"""
Advanced training script for Expert-Sliced GPU Scheduling MoE.
Demonstrates all features: Triton kernels, CUDA graphs, dynamic slicing, and energy monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_gpu.model import AdvancedMoELayer
from moe_gpu.gpu_slice_manager import SliceAllocationPolicy
from moe_gpu.benchmark import MoEBenchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
config = {
    'input_dim': 512,
    'expert_dim': 512,
    'hidden_dim': 1024,
    'num_experts': 16,
    'top_k': 2,
    'batch_size': 128,
    'num_epochs': 20,
    'learning_rate': 1e-3,
    'total_slices': 8,
    'min_slices': 1,
    'max_slices': 4,
    'update_interval': 50,
    'use_triton': True,
    'use_cuda_graphs': True,
    'enable_energy_monitoring': True,
    'allocation_policy': 'dynamic'
}


def generate_synthetic_data(num_samples: int, input_dim: int, num_patterns: int = 8) -> torch.Tensor:
    """
    Generate synthetic data with distinct patterns for different experts to specialize in.
    """
    patterns = [
        lambda x: torch.sin(x * 0.5),
        lambda x: torch.cos(x * 0.3),
        lambda x: x ** 2 * 0.1,
        lambda x: torch.sqrt(torch.abs(x) + 1e-6),
        lambda x: torch.exp(-x ** 2 * 0.1),
        lambda x: torch.sigmoid(x * 2),
        lambda x: torch.tanh(x),
        lambda x: torch.relu(x) * 0.5
    ]
    
    data = torch.randn(num_samples, input_dim)
    
    # Apply different patterns to different segments
    pattern_size = input_dim // len(patterns)
    for i, pattern in enumerate(patterns):
        start = i * pattern_size
        end = (i + 1) * pattern_size if i < len(patterns) - 1 else input_dim
        data[:, start:end] = pattern(data[:, start:end])
    
    return data


def train_advanced_moe():
    """Train the advanced MoE model with all optimizations."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type != 'cuda':
        logger.warning("CUDA not available. Some optimizations will be disabled.")
        config['use_triton'] = False
        config['use_cuda_graphs'] = False
        config['enable_energy_monitoring'] = False
    
    # Map allocation policy string to enum
    policy_map = {
        'static': SliceAllocationPolicy.STATIC,
        'dynamic': SliceAllocationPolicy.DYNAMIC,
        'proportional': SliceAllocationPolicy.PROPORTIONAL,
        'adaptive': SliceAllocationPolicy.ADAPTIVE
    }
    allocation_policy = policy_map.get(config['allocation_policy'], SliceAllocationPolicy.DYNAMIC)
    
    # Initialize model
    logger.info("Initializing Advanced MoE model...")
    model = AdvancedMoELayer(
        input_dim=config['input_dim'],
        expert_dim=config['expert_dim'],
        hidden_dim=config['hidden_dim'],
        num_experts=config['num_experts'],
        top_k=config['top_k'],
        total_slices=config['total_slices'],
        use_triton=config['use_triton'],
        use_cuda_graphs=config['use_cuda_graphs'],
        enable_energy_monitoring=config['enable_energy_monitoring'],
        allocation_policy=allocation_policy
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Generate synthetic data
    logger.info("Generating training data...")
    X_train = generate_synthetic_data(20000, config['input_dim'])
    y_train = X_train.clone()  # Autoencoder task
    
    X_val = generate_synthetic_data(2000, config['input_dim'])
    y_val = X_val.clone()
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Training metrics
    train_losses = []
    val_losses = []
    expert_utilization_history = []
    slice_allocation_history = []
    energy_history = []
    
    logger.info("Starting training...")
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_stats = []
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output, stats = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_stats.append(stats)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                val_loss += criterion(output, target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Get performance statistics
        perf_stats = model.get_performance_stats()
        
        # Record expert utilization
        expert_util = perf_stats['profiler_stats']['expert_utilization']
        expert_utilization_history.append(expert_util)
        
        # Record slice allocations
        slice_stats = perf_stats['slice_stats']
        slice_allocation_history.append(slice_stats)
        
        # Record energy metrics
        if config['enable_energy_monitoring'] and 'energy_stats' in perf_stats:
            energy_stats = perf_stats['energy_stats']
            energy_history.append(energy_stats)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} - "
                   f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        logger.info(f"  GPU Slices: {slice_stats['allocated_slices']}/{slice_stats['total_slices']} allocated, "
                   f"Avg Utilization: {slice_stats['avg_utilization']:.2f}")
        
        if config['enable_energy_monitoring'] and energy_history:
            latest_energy = energy_history[-1]
            efficiency = latest_energy.get('efficiency_metrics', {})
            logger.info(f"  Energy: {efficiency.get('avg_power_watts', 0):.2f}W, "
                       f"{efficiency.get('tokens_per_joule', 0):.2f} tokens/J")
        
        # Optimize allocations periodically
        if (epoch + 1) % 5 == 0:
            model.optimize_allocations()
            logger.info("  Optimized GPU slice allocations")
        
        # Update learning rate
        scheduler.step()
    
    # Final performance statistics
    logger.info("\n" + "="*80)
    logger.info("Training Complete - Final Statistics")
    logger.info("="*80)
    
    final_stats = model.get_performance_stats()
    
    logger.info(f"\nExpert Utilization:")
    expert_util = final_stats['profiler_stats']['expert_utilization']
    for expert_id, util in sorted(expert_util.items()):
        logger.info(f"  Expert {expert_id}: {util:.2f} tokens/sec")
    
    logger.info(f"\nGPU Slice Statistics:")
    slice_stats = final_stats['slice_stats']
    logger.info(f"  Total Slices: {slice_stats['total_slices']}")
    logger.info(f"  Allocated: {slice_stats['allocated_slices']}")
    logger.info(f"  Average Utilization: {slice_stats['avg_utilization']:.2f}")
    logger.info(f"  Reallocations: {slice_stats['reallocation_count']}")
    
    if config['use_cuda_graphs']:
        logger.info(f"\nCUDA Graph Statistics:")
        graph_stats = final_stats['graph_stats']
        logger.info(f"  Captured Graphs: {graph_stats['num_graphs']}")
        logger.info(f"  Expert IDs: {graph_stats['expert_ids']}")
    
    if config['enable_energy_monitoring'] and 'energy_stats' in final_stats:
        logger.info(f"\nEnergy Efficiency:")
        energy_stats = final_stats['energy_stats']
        efficiency = energy_stats['efficiency_metrics']
        logger.info(f"  Total Energy: {efficiency['total_energy_joules']:.2f} J")
        logger.info(f"  Average Power: {efficiency['avg_power_watts']:.2f} W")
        logger.info(f"  Tokens/Joule: {efficiency['tokens_per_joule']:.2f}")
        logger.info(f"  Tokens/Second: {efficiency['tokens_per_second']:.2f}")
        
        logger.info(f"\n  Expert Energy Comparison:")
        expert_comparison = energy_stats['expert_comparison']
        for comp in expert_comparison[:5]:  # Top 5 most efficient
            logger.info(f"    Expert {comp['expert_id']}: "
                       f"{comp['energy_per_token']:.6f} J/token, "
                       f"{comp['throughput']:.2f} tokens/sec")
    
    logger.info("="*80 + "\n")
    
    # Plot training curves
    plot_training_results(
        train_losses, val_losses,
        expert_utilization_history,
        slice_allocation_history,
        energy_history
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_stats': final_stats
    }, 'advanced_moe_model.pth')
    logger.info("Model saved as 'advanced_moe_model.pth'")
    
    # Save statistics
    with open('training_stats.json', 'w') as f:
        json.dump({
            'config': config,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_stats': {k: v for k, v in final_stats.items() if k != 'energy_stats'}  # Exclude non-serializable
        }, f, indent=2)
    
    return model, final_stats


def plot_training_results(train_losses, val_losses, expert_util_history, slice_history, energy_history):
    """Plot comprehensive training results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training and Validation Loss
    axes[0, 0].plot(train_losses, label='Training Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: GPU Slice Utilization
    if slice_history:
        allocated = [s['allocated_slices'] for s in slice_history]
        avg_util = [s['avg_utilization'] for s in slice_history]
        
        ax2 = axes[0, 1]
        ax2.plot(allocated, label='Allocated Slices', linewidth=2, color='blue')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Allocated Slices', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.grid(True, alpha=0.3)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(avg_util, label='Avg Utilization', linewidth=2, color='red')
        ax2_twin.set_ylabel('Utilization', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        ax2.set_title('GPU Slice Allocation and Utilization')
    
    # Plot 3: Expert Utilization Heatmap
    if expert_util_history:
        # Convert to matrix
        num_epochs = len(expert_util_history)
        num_experts = max(max(util.keys()) for util in expert_util_history if util) + 1
        
        util_matrix = np.zeros((num_epochs, num_experts))
        for epoch, util_dict in enumerate(expert_util_history):
            for expert_id, util_val in util_dict.items():
                util_matrix[epoch, expert_id] = util_val
        
        im = axes[1, 0].imshow(util_matrix.T, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Expert ID')
        axes[1, 0].set_title('Expert Utilization Over Time')
        plt.colorbar(im, ax=axes[1, 0], label='Tokens/sec')
    
    # Plot 4: Energy Efficiency
    if energy_history:
        power = [e['efficiency_metrics']['avg_power_watts'] for e in energy_history if 'efficiency_metrics' in e]
        tokens_per_joule = [e['efficiency_metrics']['tokens_per_joule'] for e in energy_history if 'efficiency_metrics' in e]
        
        if power and tokens_per_joule:
            ax4 = axes[1, 1]
            ax4.plot(power, label='Avg Power (W)', linewidth=2, color='orange')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Power (W)', color='orange')
            ax4.tick_params(axis='y', labelcolor='orange')
            ax4.grid(True, alpha=0.3)
            
            ax4_twin = ax4.twinx()
            ax4_twin.plot(tokens_per_joule, label='Tokens/Joule', linewidth=2, color='green')
            ax4_twin.set_ylabel('Tokens/Joule', color='green')
            ax4_twin.tick_params(axis='y', labelcolor='green')
            
            ax4.set_title('Energy Efficiency Over Time')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Training plots saved as 'training_results.png'")


if __name__ == "__main__":
    train_advanced_moe()
