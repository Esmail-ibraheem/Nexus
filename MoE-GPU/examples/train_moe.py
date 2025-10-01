import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_gpu.model import MoELayer, GPUSliceManager
from moe_gpu.profiler import ExpertProfiler, GPUSliceOptimizer

# Configuration
config = {
    'input_dim': 64,
    'expert_dim': 64,
    'hidden_dim': 128,
    'num_experts': 8,
    'top_k': 2,
    'batch_size': 256,
    'num_epochs': 10,
    'learning_rate': 1e-3,
    'total_slices': 8,
    'min_slices': 1,
    'max_slices': 4,
    'update_interval': 100
}

def generate_data(num_samples: int, input_dim: int) -> torch.Tensor:
    """Generate synthetic data with different patterns."""
    # Create patterns that different experts can specialize in
    patterns = [
        lambda x: torch.sin(x * 0.1),
        lambda x: torch.cos(x * 0.2),
        lambda x: x ** 2,
        lambda x: torch.sqrt(torch.abs(x) + 1e-6),
        lambda x: torch.exp(-x ** 2),
        lambda x: torch.sigmoid(x),
        lambda x: torch.tanh(x),
        lambda x: torch.relu(x)
    ]
    
    # Generate random data
    data = torch.randn(num_samples, input_dim)
    
    # Apply patterns
    pattern_size = input_dim // len(patterns)
    for i, pattern in enumerate(patterns):
        start = i * pattern_size
        end = (i + 1) * pattern_size if i < len(patterns) - 1 else input_dim
        data[:, start:end] = pattern(data[:, start:end])
    
    return data

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize profiler and optimizer
    profiler = ExpertProfiler()
    slice_optimizer = GPUSliceOptimizer(
        profiler=profiler,
        total_slices=config['total_slices'],
        min_slices=config['min_slices'],
        max_slices=config['max_slices'],
        update_interval=config['update_interval']
    )
    
    # Initialize GPU slice manager
    slice_manager = GPUSliceManager(total_slices=config['total_slices'])
    
    # Initialize model
    model = MoELayer(
        input_dim=config['input_dim'],
        expert_dim=config['expert_dim'],
        hidden_dim=config['hidden_dim'],
        num_experts=config['num_experts'],
        top_k=config['top_k'],
        gpu_slice_manager=slice_manager
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Generate synthetic data
    print("Generating training data...")
    X_train = generate_data(10000, config['input_dim'])
    y_train = X_train  # Autoencoder-like task
    
    X_val = generate_data(1000, config['input_dim'])
    y_val = X_val
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    slice_allocations = []
    
    print("Starting training...")
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        
        # Training phase
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Get slice allocation recommendations
            allocations = slice_optimizer.step()
            if allocations:
                print(f"\nNew slice allocations: {allocations}")
                slice_allocations.append(allocations)
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} - "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
    
    # Save model
    torch.save(model.state_dict(), 'moe_model.pth')
    print("Training complete. Model saved as 'moe_model.pth'")

if __name__ == "__main__":
    train()
