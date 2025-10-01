# Quick Start Guide - Expert-Sliced GPU Scheduling

This guide will help you get started with the Expert-Sliced GPU Scheduling system in 5 minutes.

## Prerequisites

- NVIDIA GPU (A100, H100, or RTX 3090+)
- Python 3.10+
- CUDA 12.0+

## Installation (2 minutes)

```bash
# Clone and setup
git clone https://github.com/yourusername/moe-gpu-scheduling.git
cd moe-gpu-scheduling
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run Your First Benchmark (3 minutes)

### Option 1: Quick Benchmark (Recommended for first run)

```bash
python examples/run_benchmark.py
# Select option 1 when prompted
```

This will:
- Compare baseline vs. optimized MoE
- Test 3 batch sizes (64, 128, 256)
- Run 50 iterations per configuration
- Display results in a formatted table
- Save results to `benchmark_results.json`

**Expected runtime:** ~2-3 minutes on A100

### Option 2: Train a Model

```bash
python examples/train_advanced_moe.py
```

This will:
- Train an MoE model with all optimizations
- Show real-time GPU utilization and energy metrics
- Generate training plots
- Save model and statistics

**Expected runtime:** ~5-10 minutes for 20 epochs

## Understanding the Results

### Benchmark Output

```
Batch Size   Baseline (ms)   Optimized (ms)  Speedup    Improvement    
------------------------------------------------------------------------
64           2.76            1.22            2.26×      126.23%
128          4.89            2.11            2.32×      131.75%
256          8.34            3.61            2.31×      130.75%
```

**What this means:**
- **Speedup**: How many times faster the optimized version is
- **Improvement**: Percentage increase in throughput
- **2.3× speedup** = Processing 2.3× more tokens per second

### Training Output

```
Epoch 1/20 - Train Loss: 0.234567, Val Loss: 0.245678
  GPU Slices: 6/8 allocated, Avg Utilization: 0.73
  Energy: 245.32W, 1234.56 tokens/J
```

**What this means:**
- **GPU Slices**: 6 out of 8 GPU partitions are actively used
- **Utilization 0.73**: 73% of GPU is doing useful work (vs ~28% baseline)
- **tokens/J**: Energy efficiency metric (higher is better)

## Visualize Results

```bash
python scripts/visualize_results.py --results benchmark_results.json
```

This generates publication-quality plots in the `plots/` directory.

## Next Steps

### 1. Customize Configuration

Edit the config in `examples/train_advanced_moe.py`:

```python
config = {
    'num_experts': 16,      # Try 8, 16, 32
    'top_k': 2,             # Number of experts per token
    'batch_size': 128,      # Adjust based on GPU memory
    'use_triton': True,     # Triton kernel optimization
    'use_cuda_graphs': True # CUDA graph optimization
}
```

### 2. Run Full Benchmark

```bash
python examples/run_benchmark.py
# Select option 2 for full benchmark
```

This tests 6 batch sizes with 100 iterations each (~10 minutes).

### 3. Compare Allocation Policies

Test different GPU slicing strategies:

```python
from moe_gpu import AdvancedMoELayer, SliceAllocationPolicy

# Try different policies
for policy in [SliceAllocationPolicy.STATIC, 
               SliceAllocationPolicy.DYNAMIC,
               SliceAllocationPolicy.PROPORTIONAL]:
    model = AdvancedMoELayer(..., allocation_policy=policy)
    # Benchmark and compare
```

## Common Use Cases

### Use Case 1: Maximum Throughput

```python
model = AdvancedMoELayer(
    num_experts=16,
    top_k=2,
    use_triton=True,
    use_cuda_graphs=True,
    allocation_policy=SliceAllocationPolicy.DYNAMIC
)
```

### Use Case 2: Energy Efficiency

```python
model = AdvancedMoELayer(
    num_experts=8,
    top_k=2,
    enable_energy_monitoring=True,
    allocation_policy=SliceAllocationPolicy.PROPORTIONAL
)
```

### Use Case 3: Research/Analysis

```python
model = AdvancedMoELayer(
    num_experts=32,
    top_k=4,
    use_triton=True,
    use_cuda_graphs=True,
    enable_energy_monitoring=True
)

# Get detailed statistics
stats = model.get_performance_stats()
print(stats['energy_stats'])
print(stats['slice_stats'])
```

## Troubleshooting

### Problem: "CUDA out of memory"

**Solution:** Reduce batch size or number of experts

```bash
# In train_advanced_moe.py, change:
config['batch_size'] = 64  # Instead of 128
config['num_experts'] = 8  # Instead of 16
```

### Problem: "Triton not found"

**Solution:** Install Triton or disable it

```bash
pip install triton>=2.0.0
# OR
model = AdvancedMoELayer(..., use_triton=False)
```

### Problem: "NVML initialization failed"

**Solution:** Disable energy monitoring

```python
model = AdvancedMoELayer(..., enable_energy_monitoring=False)
```

## Performance Tips

1. **Batch Size**: Larger batches (≥128) show better speedup
2. **Expert Count**: More experts (≥16) benefit more from slicing
3. **Top-K**: k=2 is optimal for most cases
4. **Warmup**: First few iterations are slower (CUDA graph capture)

## What's Happening Under the Hood?

1. **Routing**: Triton kernel selects top-k experts per token
2. **Profiling**: System tracks which experts are used most
3. **Slicing**: GPU resources allocated based on expert popularity
4. **Execution**: Experts run in parallel on dedicated streams
5. **Graphs**: Frequently-used patterns captured for fast replay

## Getting Help

- **Documentation**: See full README.md
- **Research Paper**: See paper/research_paper.md
- **Issues**: GitHub Issues
- **Examples**: Check examples/ directory

## Quick Reference

### Run Commands

```bash
# Quick benchmark
python examples/run_benchmark.py

# Train model
python examples/train_advanced_moe.py

# Generate plots
python scripts/visualize_results.py

# Custom benchmark
python -c "from moe_gpu.benchmark import run_quick_benchmark; run_quick_benchmark()"
```

### Import Examples

```python
# Basic usage
from moe_gpu import AdvancedMoELayer

# Advanced usage
from moe_gpu import (
    AdvancedMoELayer,
    SliceAllocationPolicy,
    EnergyMonitor,
    MoEBenchmark
)
```

## Expected Performance

On NVIDIA A100:
- **Speedup**: 2.3-2.4× vs baseline
- **GPU Utilization**: 73% (vs 28% baseline)
- **Energy Efficiency**: 35-45% improvement

Your results may vary based on:
- GPU model (A100 vs H100 vs RTX)
- Model configuration (experts, dimensions)
- Batch size
- CUDA/driver versions

---

**Ready to dive deeper?** Check out the full [README.md](README.md) and [research paper](paper/research_paper.md)!
