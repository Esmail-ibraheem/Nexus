import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch

@dataclass
class ExpertStats:
    call_count: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    last_used: float = 0.0
    
class ExpertProfiler:
    """Profiles expert usage patterns to inform GPU slicing decisions."""
    def __init__(self, window_size: int = 100):
        self.stats: Dict[int, ExpertStats] = defaultdict(ExpertStats)
        self.window_size = window_size
        self.recent_calls = deque(maxlen=window_size)
        
    def record_expert_call(
        self, 
        expert_id: int, 
        num_tokens: int,
        execution_time: float
    ) -> None:
        """Record a call to an expert."""
        stats = self.stats[expert_id]
        stats.call_count += 1
        stats.total_tokens += num_tokens
        stats.total_time += execution_time
        stats.last_used = time.time()
        
        self.recent_calls.append((expert_id, num_tokens, execution_time))
    
    def get_expert_utilization(self) -> Dict[int, float]:
        """Calculate the utilization of each expert."""
        util = {}
        for expert_id, stats in self.stats.items():
            if stats.call_count > 0:
                # Utilization is tokens per second, normalized by expert capacity
                util[expert_id] = stats.total_tokens / (stats.total_time + 1e-6)
            else:
                util[expert_id] = 0.0
        return util
    
    def get_hot_experts(self, threshold: float = 0.8) -> List[int]:
        """Get experts with utilization above a threshold."""
        util = self.get_expert_utilization()
        return [eid for eid, u in util.items() if u > threshold]
    
    def get_cold_experts(self, threshold: float = 0.2) -> List[int]:
        """Get experts with utilization below a threshold."""
        util = self.get_expert_utilization()
        return [eid for eid, u in util.items() if u < threshold]
    
    def get_slice_recommendations(
        self, 
        total_slices: int,
        min_slices: int = 1,
        max_slices: int = 4
    ) -> Dict[int, int]:
        """Recommend number of slices for each expert based on utilization."""
        util = self.get_expert_utilization()
        if not util:
            return {}
            
        # Normalize utilization to [0, 1]
        max_util = max(util.values()) if max(util.values()) > 0 else 1.0
        normal_util = {eid: u / max_util for eid, u in util.items()}
        
        # Sort experts by utilization
        sorted_experts = sorted(normal_util.items(), key=lambda x: -x[1])
        
        # Allocate slices based on utilization
        remaining_slices = total_slices
        allocations = {}
        
        # First pass: allocate minimum slices to all experts
        for eid, _ in sorted_experts:
            allocations[eid] = min_slices
            remaining_slices -= min_slices
            
        # Second pass: distribute remaining slices to high-utilization experts
        if remaining_slices > 0:
            for eid, u in sorted_experts:
                if remaining_slices <= 0:
                    break
                    
                # Allocate more slices to higher utilization experts
                additional = min(
                    max_slices - min_slices,
                    int(u * (remaining_slices + 1))
                )
                
                if additional > 0:
                    allocations[eid] += additional
                    remaining_slices -= additional
        
        return allocations

class GPUSliceOptimizer:
    """Optimizes GPU slice allocation based on expert profiling."""
    def __init__(
        self, 
        profiler: ExpertProfiler,
        total_slices: int = 8,
        update_interval: int = 100,
        min_slices: int = 1,
        max_slices: int = 4
    ):
        self.profiler = profiler
        self.total_slices = total_slices
        self.update_interval = update_interval
        self.min_slices = min_slices
        self.max_slices = max_slices
        self.step_count = 0
        
    def step(self) -> Dict[int, int]:
        """Update slice allocations based on current profiling data."""
        self.step_count += 1
        
        if self.step_count % self.update_interval != 0:
            return {}
            
        return self.profiler.get_slice_recommendations(
            self.total_slices,
            self.min_slices,
            self.max_slices
        )
