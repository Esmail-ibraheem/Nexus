"""
Advanced Mixture of Experts implementation with GPU slicing, CUDA graphs, and Triton kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np
import time
import logging

from .triton_kernels import TritonExpertOps
from .cuda_graph_manager import CUDAGraphManager, StreamManager, BatchScheduler
from .gpu_slice_manager import AdvancedGPUSliceManager, SliceAllocationPolicy
from .profiler import ExpertProfiler, GPUSliceOptimizer
from .energy_monitor import EnergyMonitor

logger = logging.getLogger(__name__)


class Expert(nn.Module):
    """Expert network with configurable architecture."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdvancedMoELayer(nn.Module):
    """
    Advanced Mixture of Experts layer with:
    - Dynamic GPU slicing
    - CUDA graph optimization
    - Triton kernel acceleration
    - Stream-based parallel execution
    - Energy monitoring
    """
    
    def __init__(
        self,
        input_dim: int,
        expert_dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.0,
        total_slices: int = 8,
        use_triton: bool = True,
        use_cuda_graphs: bool = True,
        enable_energy_monitoring: bool = True,
        allocation_policy: SliceAllocationPolicy = SliceAllocationPolicy.DYNAMIC
    ):
        super().__init__()
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.use_triton = use_triton
        self.use_cuda_graphs = use_cuda_graphs
        
        # Router network
        self.router = nn.Linear(input_dim, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, expert_dim) 
            for _ in range(num_experts)
        ])
        
        # GPU slice manager
        self.slice_manager = AdvancedGPUSliceManager(
            total_slices=total_slices,
            allocation_policy=allocation_policy
        )
        
        # CUDA graph manager
        self.graph_manager = CUDAGraphManager(enable_graphs=use_cuda_graphs)
        
        # Stream manager
        self.stream_manager = StreamManager(num_streams=total_slices)
        
        # Batch scheduler
        self.batch_scheduler = BatchScheduler(
            stream_manager=self.stream_manager,
            graph_manager=self.graph_manager
        )
        
        # Profiler
        self.profiler = ExpertProfiler()
        
        # Slice optimizer
        self.slice_optimizer = GPUSliceOptimizer(
            profiler=self.profiler,
            total_slices=total_slices
        )
        
        # Energy monitor
        self.energy_monitor = EnergyMonitor() if enable_energy_monitoring else None
        if self.energy_monitor:
            self.energy_monitor.start_monitoring()
        
        # Triton ops
        self.triton_ops = TritonExpertOps() if use_triton else None
        
        # Statistics
        self.forward_count = 0
        self.routing_stats = {'expert_counts': [0] * num_experts}
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with advanced GPU scheduling.
        
        Args:
            x: [batch_size, input_dim] input tensor
            
        Returns:
            output: [batch_size, expert_dim] output tensor
            stats: Dictionary of execution statistics
        """
        batch_size = x.size(0)
        device = x.device
        start_time = time.time()
        
        # Sample energy metrics
        if self.energy_monitor:
            self.energy_monitor.sample_metrics()
        
        # Routing
        logits = self.router(x)
        
        if self.use_triton and self.triton_ops and x.is_cuda:
            # Use Triton kernel for routing
            expert_ids, expert_weights, expert_counts = self.triton_ops.expert_routing(
                logits, self.top_k
            )
        else:
            # Standard PyTorch routing
            probs = F.softmax(logits, dim=-1)
            expert_weights, expert_ids = torch.topk(probs, self.top_k, dim=-1)
            expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-6)
            expert_counts = torch.bincount(expert_ids.flatten(), minlength=self.num_experts)
        
        # Initialize output
        output = torch.zeros(batch_size, self.expert_dim, device=device, dtype=x.dtype)
        
        # Process each expert
        expert_outputs = {}
        expert_timings = {}
        
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_ids == expert_idx).any(dim=-1)
            
            if not expert_mask.any():
                continue
            
            expert_start = time.time()
            
            # Get tokens for this expert
            expert_input = x[expert_mask]
            num_tokens = expert_input.shape[0]
            
            # Allocate GPU slices based on profiling
            slice_recommendations = self.slice_optimizer.step()
            required_slices = slice_recommendations.get(expert_idx, 1)
            
            try:
                allocation = self.slice_manager.allocate_slices(
                    expert_idx,
                    required_slices,
                    priority=int(expert_counts[expert_idx].item())
                )
                slice_id = allocation.slice_ids[0]
            except RuntimeError:
                # Fallback to minimal allocation
                allocation = self.slice_manager.allocate_slices(expert_idx, 1, priority=0)
                slice_id = allocation.slice_ids[0]
            
            # Execute expert with scheduling
            expert_output = self.batch_scheduler.schedule_expert_batch(
                expert_id=expert_idx,
                expert_module=self.experts[expert_idx],
                input_tensor=expert_input,
                slice_id=slice_id,
                use_graph=self.use_cuda_graphs
            )
            
            expert_outputs[expert_idx] = expert_output
            
            expert_end = time.time()
            expert_timings[expert_idx] = expert_end - expert_start
            
            # Record profiling data
            self.profiler.record_expert_call(
                expert_idx,
                num_tokens,
                expert_timings[expert_idx]
            )
            
            # Record energy data
            if self.energy_monitor:
                self.energy_monitor.record_expert_execution(
                    expert_idx,
                    num_tokens,
                    expert_start,
                    expert_end
                )
        
        # Synchronize all streams
        self.stream_manager.synchronize_all()
        
        # Combine expert outputs
        for expert_idx, expert_output in expert_outputs.items():
            expert_mask = (expert_ids == expert_idx).any(dim=-1)
            
            # Get routing weights for this expert
            for k in range(self.top_k):
                k_mask = (expert_ids[:, k] == expert_idx)
                if k_mask.any():
                    weights = expert_weights[k_mask, k].unsqueeze(-1)
                    output[k_mask] += expert_output[:k_mask.sum()] * weights
        
        end_time = time.time()
        
        # Collect statistics
        stats = {
            'forward_time': end_time - start_time,
            'expert_timings': expert_timings,
            'expert_counts': expert_counts.cpu().numpy().tolist(),
            'slice_allocations': {
                eid: alloc.slice_ids 
                for eid, alloc in self.slice_manager.expert_allocations.items()
            },
            'batch_size': batch_size
        }
        
        self.forward_count += 1
        
        return output, stats
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        stats = {
            'forward_count': self.forward_count,
            'profiler_stats': {
                'expert_utilization': self.profiler.get_expert_utilization(),
                'hot_experts': self.profiler.get_hot_experts(),
                'cold_experts': self.profiler.get_cold_experts()
            },
            'slice_stats': self.slice_manager.get_stats(),
            'stream_stats': self.stream_manager.get_stats(),
            'graph_stats': self.graph_manager.get_stats(),
            'batch_stats': self.batch_scheduler.get_batch_stats()
        }
        
        if self.energy_monitor:
            stats['energy_stats'] = {
                'total_energy': self.energy_monitor.get_total_energy(),
                'avg_power': self.energy_monitor.get_average_power(),
                'avg_utilization': self.energy_monitor.get_average_utilization(),
                'efficiency_metrics': self.energy_monitor.get_efficiency_metrics(),
                'expert_comparison': self.energy_monitor.get_expert_comparison()
            }
        
        return stats
    
    def optimize_allocations(self):
        """Trigger optimization of GPU slice allocations."""
        self.slice_manager.optimize_allocations()
    
    def reset_stats(self):
        """Reset all statistics."""
        self.forward_count = 0
        self.routing_stats = {'expert_counts': [0] * self.num_experts}
        self.batch_scheduler.reset_history()
        if self.energy_monitor:
            self.energy_monitor.reset()


# Legacy compatibility
class GPUSliceManager:
    """Legacy GPU slice manager for backward compatibility."""
    def __init__(self, total_slices: int = 8):
        self.total_slices = total_slices
        self.available_slices = list(range(total_slices))
        self.expert_to_slice = {}
        self.slice_to_expert = {i: None for i in range(total_slices)}
    
    def allocate_slices(self, expert_id: int, required_slices: int) -> List[int]:
        if expert_id in self.expert_to_slice:
            return self.expert_to_slice[expert_id]
            
        if len(self.available_slices) < required_slices:
            raise RuntimeError("Not enough GPU slices available")
            
        allocated = self.available_slices[:required_slices]
        self.available_slices = self.available_slices[required_slices:]
        self.expert_to_slice[expert_id] = allocated
        for s in allocated:
            self.slice_to_expert[s] = expert_id
            
        return allocated
    
    def release_slices(self, expert_id: int):
        if expert_id not in self.expert_to_slice:
            return
            
        for s in self.expert_to_slice[expert_id]:
            self.slice_to_expert[s] = None
            self.available_slices.append(s)
            
        del self.expert_to_slice[expert_id]


class MoELayer(nn.Module):
    """Legacy MoE layer for backward compatibility."""
    def __init__(
        self,
        input_dim: int,
        expert_dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.0,
        gpu_slice_manager: Optional[GPUSliceManager] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        self.router = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, expert_dim) 
            for _ in range(num_experts)
        ])
        
        self.gpu_slice_manager = gpu_slice_manager or GPUSliceManager()
        self.expert_to_device = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        output = torch.zeros(batch_size, self.expert_dim, device=x.device)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (topk_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            if expert_idx not in self.expert_to_device:
                required_slices = min(2, self.gpu_slice_manager.total_slices)
                self.gpu_slice_manager.allocate_slices(expert_idx, required_slices)
                
            expert_input = x[expert_mask]
            expert_output = self.experts[expert_idx](expert_input)
            
            expert_weights = torch.zeros(batch_size, 1, device=x.device)
            for k in range(self.top_k):
                mask = (topk_indices[:, k] == expert_idx)
                expert_weights[mask] = topk_probs[mask, k].unsqueeze(-1)
            
            output[expert_mask] += expert_output * expert_weights[expert_mask]
            
        return output
