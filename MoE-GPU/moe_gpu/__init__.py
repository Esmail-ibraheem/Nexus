"""
Expert-Sliced GPU Scheduling for Mixture of Experts (MoE)

This package provides tools for implementing and optimizing Mixture of Experts models
with dynamic GPU resource allocation, CUDA graphs, Triton kernels, and energy monitoring.
"""

from .model import (
    MoELayer, 
    GPUSliceManager, 
    Expert,
    AdvancedMoELayer
)
from .profiler import ExpertProfiler, GPUSliceOptimizer
from .triton_kernels import TritonExpertOps
from .cuda_graph_manager import CUDAGraphManager, StreamManager, BatchScheduler
from .gpu_slice_manager import AdvancedGPUSliceManager, SliceAllocationPolicy
from .energy_monitor import EnergyMonitor, PerformanceComparator

__version__ = "0.1.0"
__all__ = [
    # Core models
    'MoELayer',
    'AdvancedMoELayer',
    'Expert',
    
    # GPU management
    'GPUSliceManager',
    'AdvancedGPUSliceManager',
    'SliceAllocationPolicy',
    
    # Profiling and optimization
    'ExpertProfiler',
    'GPUSliceOptimizer',
    
    # CUDA optimization
    'TritonExpertOps',
    'CUDAGraphManager',
    'StreamManager',
    'BatchScheduler',
    
    # Monitoring
    'EnergyMonitor',
    'PerformanceComparator'
]
