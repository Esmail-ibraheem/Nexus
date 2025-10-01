"""
Comprehensive benchmarking suite for Expert-Sliced GPU Scheduling.
Compares baseline MoE vs. optimized implementation across multiple metrics.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple
import json
import logging

from .model import MoELayer, AdvancedMoELayer
from .gpu_slice_manager import SliceAllocationPolicy
from .energy_monitor import EnergyMonitor, PerformanceComparator

logger = logging.getLogger(__name__)


class MoEBenchmark:
    """
    Benchmark suite for comparing MoE implementations.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        expert_dim: int = 512,
        hidden_dim: int = 1024,
        num_experts: int = 8,
        top_k: int = 2,
        batch_sizes: List[int] = None,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ):
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.batch_sizes = batch_sizes or [32, 64, 128, 256, 512]
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def create_baseline_model(self) -> MoELayer:
        """Create baseline MoE model."""
        model = MoELayer(
            input_dim=self.input_dim,
            expert_dim=self.expert_dim,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k
        ).to(self.device)
        return model
    
    def create_optimized_model(
        self,
        use_triton: bool = True,
        use_cuda_graphs: bool = True,
        enable_energy_monitoring: bool = True,
        allocation_policy: SliceAllocationPolicy = SliceAllocationPolicy.DYNAMIC
    ) -> AdvancedMoELayer:
        """Create optimized MoE model with all features."""
        model = AdvancedMoELayer(
            input_dim=self.input_dim,
            expert_dim=self.expert_dim,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            use_triton=use_triton,
            use_cuda_graphs=use_cuda_graphs,
            enable_energy_monitoring=enable_energy_monitoring,
            allocation_policy=allocation_policy
        ).to(self.device)
        return model
    
    def benchmark_model(
        self,
        model: nn.Module,
        batch_size: int,
        model_name: str
    ) -> Dict:
        """
        Benchmark a single model configuration.
        
        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"Benchmarking {model_name} with batch_size={batch_size}")
        
        # Generate random input
        x = torch.randn(batch_size, self.input_dim, device=self.device)
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                if isinstance(model, AdvancedMoELayer):
                    _ = model(x)
                else:
                    _ = model(x)
        
        # Synchronize before timing
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        timings = []
        start_events = []
        end_events = []
        
        with torch.no_grad():
            for i in range(self.num_iterations):
                if self.device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    if isinstance(model, AdvancedMoELayer):
                        output, stats = model(x)
                    else:
                        output = model(x)
                    end_event.record()
                    
                    start_events.append(start_event)
                    end_events.append(end_event)
                else:
                    start_time = time.time()
                    if isinstance(model, AdvancedMoELayer):
                        output, stats = model(x)
                    else:
                        output = model(x)
                    end_time = time.time()
                    timings.append(end_time - start_time)
        
        # Synchronize and collect timings
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            for start_event, end_event in zip(start_events, end_events):
                timings.append(start_event.elapsed_time(end_event) / 1000.0)  # Convert to seconds
        
        # Calculate statistics
        timings = np.array(timings)
        mean_time = np.mean(timings)
        std_time = np.std(timings)
        min_time = np.min(timings)
        max_time = np.max(timings)
        
        throughput = batch_size / mean_time  # tokens per second
        
        metrics = {
            'model_name': model_name,
            'batch_size': batch_size,
            'mean_time': mean_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'throughput': throughput,
            'tokens_per_second': throughput
        }
        
        # Get additional stats from AdvancedMoELayer
        if isinstance(model, AdvancedMoELayer):
            perf_stats = model.get_performance_stats()
            metrics.update({
                'slice_stats': perf_stats.get('slice_stats', {}),
                'energy_stats': perf_stats.get('energy_stats', {}),
                'graph_stats': perf_stats.get('graph_stats', {}),
                'stream_stats': perf_stats.get('stream_stats', {})
            })
        
        return metrics
    
    def run_comparison(self) -> Dict:
        """
        Run comprehensive comparison between baseline and optimized models.
        
        Returns:
            Dictionary containing all benchmark results
        """
        logger.info("Starting comprehensive benchmark...")
        
        results = {
            'config': {
                'input_dim': self.input_dim,
                'expert_dim': self.expert_dim,
                'hidden_dim': self.hidden_dim,
                'num_experts': self.num_experts,
                'top_k': self.top_k,
                'device': str(self.device)
            },
            'baseline': {},
            'optimized': {},
            'comparison': {}
        }
        
        # Benchmark baseline
        logger.info("Benchmarking baseline model...")
        baseline_model = self.create_baseline_model()
        
        for batch_size in self.batch_sizes:
            metrics = self.benchmark_model(baseline_model, batch_size, 'Baseline')
            results['baseline'][batch_size] = metrics
        
        del baseline_model
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Benchmark optimized
        logger.info("Benchmarking optimized model...")
        optimized_model = self.create_optimized_model()
        
        for batch_size in self.batch_sizes:
            metrics = self.benchmark_model(optimized_model, batch_size, 'Optimized')
            results['optimized'][batch_size] = metrics
        
        # Calculate improvements
        for batch_size in self.batch_sizes:
            baseline_metrics = results['baseline'][batch_size]
            optimized_metrics = results['optimized'][batch_size]
            
            speedup = baseline_metrics['mean_time'] / optimized_metrics['mean_time']
            throughput_improvement = (
                (optimized_metrics['throughput'] - baseline_metrics['throughput']) / 
                baseline_metrics['throughput'] * 100
            )
            
            results['comparison'][batch_size] = {
                'speedup': speedup,
                'throughput_improvement_percent': throughput_improvement,
                'baseline_throughput': baseline_metrics['throughput'],
                'optimized_throughput': optimized_metrics['throughput']
            }
        
        self.results = results
        return results
    
    def print_results(self):
        """Print benchmark results in a formatted table."""
        if not self.results:
            logger.warning("No results to print. Run benchmark first.")
            return
        
        print("\n" + "="*80)
        print("MoE Benchmark Results")
        print("="*80)
        
        print(f"\nConfiguration:")
        print(f"  Input dim: {self.input_dim}")
        print(f"  Expert dim: {self.expert_dim}")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Num experts: {self.num_experts}")
        print(f"  Top-k: {self.top_k}")
        print(f"  Device: {self.results['config']['device']}")
        
        print(f"\n{'Batch Size':<12} {'Baseline (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10} {'Improvement':<15}")
        print("-"*80)
        
        for batch_size in self.batch_sizes:
            baseline = self.results['baseline'][batch_size]
            optimized = self.results['optimized'][batch_size]
            comparison = self.results['comparison'][batch_size]
            
            print(f"{batch_size:<12} "
                  f"{baseline['mean_time']*1000:<15.2f} "
                  f"{optimized['mean_time']*1000:<15.2f} "
                  f"{comparison['speedup']:<10.2f}x "
                  f"{comparison['throughput_improvement_percent']:<15.2f}%")
        
        print("="*80)
        
        # Print energy stats if available
        if 'energy_stats' in self.results['optimized'][self.batch_sizes[0]]:
            print("\nEnergy Efficiency:")
            for batch_size in self.batch_sizes:
                energy_stats = self.results['optimized'][batch_size].get('energy_stats', {})
                if energy_stats:
                    efficiency = energy_stats.get('efficiency_metrics', {})
                    print(f"  Batch {batch_size}: "
                          f"{efficiency.get('tokens_per_joule', 0):.2f} tokens/J, "
                          f"{efficiency.get('avg_power_watts', 0):.2f}W avg")
        
        print("="*80 + "\n")
    
    def save_results(self, filepath: str):
        """Save benchmark results to JSON file."""
        if not self.results:
            logger.warning("No results to save. Run benchmark first.")
            return
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")


def run_quick_benchmark():
    """Run a quick benchmark with default settings."""
    benchmark = MoEBenchmark(
        input_dim=256,
        expert_dim=256,
        hidden_dim=512,
        num_experts=8,
        top_k=2,
        batch_sizes=[64, 128, 256],
        num_iterations=50,
        warmup_iterations=5
    )
    
    results = benchmark.run_comparison()
    benchmark.print_results()
    benchmark.save_results('benchmark_results.json')
    
    return results


def run_full_benchmark():
    """Run a comprehensive benchmark with multiple configurations."""
    benchmark = MoEBenchmark(
        input_dim=512,
        expert_dim=512,
        hidden_dim=2048,
        num_experts=16,
        top_k=2,
        batch_sizes=[32, 64, 128, 256, 512, 1024],
        num_iterations=100,
        warmup_iterations=10
    )
    
    results = benchmark.run_comparison()
    benchmark.print_results()
    benchmark.save_results('benchmark_results_full.json')
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_quick_benchmark()
