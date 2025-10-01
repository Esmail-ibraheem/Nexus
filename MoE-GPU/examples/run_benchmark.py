"""
Run comprehensive benchmarks comparing baseline vs. optimized MoE implementations.
"""

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_gpu.benchmark import MoEBenchmark, run_quick_benchmark, run_full_benchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("\n" + "="*80)
    print("Expert-Sliced GPU Scheduling - Benchmark Suite")
    print("="*80 + "\n")
    
    print("Select benchmark mode:")
    print("1. Quick benchmark (3 batch sizes, 50 iterations)")
    print("2. Full benchmark (6 batch sizes, 100 iterations)")
    print("3. Custom benchmark")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        print("\nRunning quick benchmark...")
        results = run_quick_benchmark()
        
    elif choice == '2':
        print("\nRunning full benchmark...")
        results = run_full_benchmark()
        
    elif choice == '3':
        print("\nCustom benchmark configuration:")
        
        input_dim = int(input("Input dimension (default 512): ") or "512")
        expert_dim = int(input("Expert dimension (default 512): ") or "512")
        hidden_dim = int(input("Hidden dimension (default 1024): ") or "1024")
        num_experts = int(input("Number of experts (default 8): ") or "8")
        top_k = int(input("Top-k experts (default 2): ") or "2")
        
        batch_sizes_str = input("Batch sizes (comma-separated, default 64,128,256): ") or "64,128,256"
        batch_sizes = [int(x.strip()) for x in batch_sizes_str.split(',')]
        
        num_iterations = int(input("Number of iterations (default 100): ") or "100")
        warmup_iterations = int(input("Warmup iterations (default 10): ") or "10")
        
        benchmark = MoEBenchmark(
            input_dim=input_dim,
            expert_dim=expert_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            batch_sizes=batch_sizes,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        print("\nRunning custom benchmark...")
        results = benchmark.run_comparison()
        benchmark.print_results()
        
        save_path = input("\nSave results to (default benchmark_custom.json): ") or "benchmark_custom.json"
        benchmark.save_results(save_path)
        
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main()
