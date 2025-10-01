"""
Advanced GPU Slice Manager with MIG support.
Implements dynamic GPU resource allocation for experts based on utilization patterns.
"""

import torch
import pynvml
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SliceAllocationPolicy(Enum):
    """Policies for allocating GPU slices to experts."""
    STATIC = "static"  # Fixed allocation
    DYNAMIC = "dynamic"  # Based on utilization
    PROPORTIONAL = "proportional"  # Proportional to expert load
    ADAPTIVE = "adaptive"  # Learns from history


@dataclass
class GPUSlice:
    """Represents a GPU slice/partition."""
    slice_id: int
    sm_count: int  # Number of streaming multiprocessors
    memory_mb: int  # Memory in MB
    compute_capacity: float  # Relative compute capacity (0-1)
    assigned_expert: Optional[int] = None
    utilization: float = 0.0


@dataclass
class ExpertAllocation:
    """Tracks GPU resource allocation for an expert."""
    expert_id: int
    slice_ids: List[int]
    total_sm_count: int
    total_memory_mb: int
    stream_id: int
    priority: int = 0


class MIGManager:
    """
    Manages NVIDIA Multi-Instance GPU (MIG) partitions.
    Provides interface to query and manage MIG instances.
    """
    
    def __init__(self):
        self.mig_enabled = False
        self.mig_devices = []
        self.device_handle = None
        
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count > 0:
                self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Check if MIG is enabled
                try:
                    mig_mode = pynvml.nvmlDeviceGetMigMode(self.device_handle)
                    self.mig_enabled = mig_mode[0] == pynvml.NVML_DEVICE_MIG_ENABLE
                    
                    if self.mig_enabled:
                        self._discover_mig_devices()
                        logger.info(f"MIG enabled with {len(self.mig_devices)} instances")
                except:
                    logger.info("MIG not supported on this device")
                    
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}")
    
    def _discover_mig_devices(self):
        """Discover available MIG devices."""
        try:
            # Query MIG device count
            count = pynvml.nvmlDeviceGetMaxMigDeviceCount(self.device_handle)
            
            for i in range(count):
                try:
                    mig_device = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(
                        self.device_handle, i
                    )
                    self.mig_devices.append(mig_device)
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to discover MIG devices: {e}")
    
    def get_mig_device_info(self, mig_index: int) -> Optional[Dict]:
        """Get information about a MIG device."""
        if mig_index >= len(self.mig_devices):
            return None
        
        try:
            mig_device = self.mig_devices[mig_index]
            
            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(mig_device)
            
            return {
                'index': mig_index,
                'total_memory_mb': memory_info.total // (1024 * 1024),
                'free_memory_mb': memory_info.free // (1024 * 1024),
                'used_memory_mb': memory_info.used // (1024 * 1024)
            }
        except Exception as e:
            logger.warning(f"Failed to get MIG device info: {e}")
            return None
    
    def is_mig_enabled(self) -> bool:
        """Check if MIG is enabled."""
        return self.mig_enabled
    
    def get_mig_device_count(self) -> int:
        """Get number of MIG devices."""
        return len(self.mig_devices)
    
    def __del__(self):
        """Cleanup NVML."""
        try:
            pynvml.nvmlShutdown()
        except:
            pass


class AdvancedGPUSliceManager:
    """
    Advanced GPU slice manager with dynamic allocation and MIG support.
    """
    
    def __init__(
        self,
        total_slices: int = 8,
        allocation_policy: SliceAllocationPolicy = SliceAllocationPolicy.DYNAMIC,
        enable_mig: bool = True
    ):
        self.total_slices = total_slices
        self.allocation_policy = allocation_policy
        self.slices: List[GPUSlice] = []
        self.expert_allocations: Dict[int, ExpertAllocation] = {}
        self.allocation_history: List[Dict] = []
        
        # MIG support
        self.mig_manager = MIGManager() if enable_mig else None
        
        # Initialize slices
        self._initialize_slices()
        
        # Performance tracking
        self.slice_utilization_history: Dict[int, List[float]] = {}
        self.reallocation_count = 0
    
    def _initialize_slices(self):
        """Initialize GPU slices."""
        if self.mig_manager and self.mig_manager.is_mig_enabled():
            # Use MIG instances as slices
            self._initialize_from_mig()
        else:
            # Create virtual slices
            self._initialize_virtual_slices()
    
    def _initialize_from_mig(self):
        """Initialize slices from MIG instances."""
        mig_count = self.mig_manager.get_mig_device_count()
        
        for i in range(min(mig_count, self.total_slices)):
            mig_info = self.mig_manager.get_mig_device_info(i)
            
            if mig_info:
                slice_obj = GPUSlice(
                    slice_id=i,
                    sm_count=0,  # Would need to query from MIG
                    memory_mb=mig_info['total_memory_mb'],
                    compute_capacity=1.0 / mig_count
                )
                self.slices.append(slice_obj)
        
        logger.info(f"Initialized {len(self.slices)} slices from MIG instances")
    
    def _initialize_virtual_slices(self):
        """Initialize virtual GPU slices."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using mock slices")
            total_memory = 16000  # Mock 16GB
            sm_count = 108  # Mock SM count
        else:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            total_memory = props.total_memory // (1024 * 1024)  # Convert to MB
            sm_count = props.multi_processor_count
        
        # Divide resources among slices
        memory_per_slice = total_memory // self.total_slices
        sm_per_slice = sm_count // self.total_slices
        
        for i in range(self.total_slices):
            slice_obj = GPUSlice(
                slice_id=i,
                sm_count=sm_per_slice,
                memory_mb=memory_per_slice,
                compute_capacity=1.0 / self.total_slices
            )
            self.slices.append(slice_obj)
        
        logger.info(f"Initialized {self.total_slices} virtual GPU slices")
    
    def allocate_slices(
        self,
        expert_id: int,
        required_slices: int,
        priority: int = 0
    ) -> ExpertAllocation:
        """
        Allocate GPU slices to an expert.
        
        Args:
            expert_id: ID of the expert
            required_slices: Number of slices required
            priority: Priority level (higher = more important)
            
        Returns:
            ExpertAllocation object
        """
        # Check if already allocated
        if expert_id in self.expert_allocations:
            return self.expert_allocations[expert_id]
        
        # Find available slices
        available = [s for s in self.slices if s.assigned_expert is None]
        
        if len(available) < required_slices:
            # Try to evict low-priority experts
            available = self._evict_low_priority_experts(required_slices, priority)
            
            if len(available) < required_slices:
                raise RuntimeError(
                    f"Cannot allocate {required_slices} slices for expert {expert_id}"
                )
        
        # Select best slices based on policy
        selected_slices = self._select_slices(available, required_slices, expert_id)
        
        # Create allocation
        allocation = ExpertAllocation(
            expert_id=expert_id,
            slice_ids=[s.slice_id for s in selected_slices],
            total_sm_count=sum(s.sm_count for s in selected_slices),
            total_memory_mb=sum(s.memory_mb for s in selected_slices),
            stream_id=selected_slices[0].slice_id % 8,  # Map to stream
            priority=priority
        )
        
        # Mark slices as assigned
        for slice_obj in selected_slices:
            slice_obj.assigned_expert = expert_id
        
        self.expert_allocations[expert_id] = allocation
        
        # Record allocation
        self.allocation_history.append({
            'expert_id': expert_id,
            'slice_ids': allocation.slice_ids,
            'timestamp': torch.cuda.Event(enable_timing=True)
        })
        
        logger.info(f"Allocated {required_slices} slices to expert {expert_id}")
        
        return allocation
    
    def _select_slices(
        self,
        available_slices: List[GPUSlice],
        count: int,
        expert_id: int
    ) -> List[GPUSlice]:
        """Select best slices for an expert based on allocation policy."""
        
        if self.allocation_policy == SliceAllocationPolicy.STATIC:
            # Just take first N available
            return available_slices[:count]
        
        elif self.allocation_policy == SliceAllocationPolicy.DYNAMIC:
            # Select slices with lowest recent utilization
            sorted_slices = sorted(
                available_slices,
                key=lambda s: s.utilization
            )
            return sorted_slices[:count]
        
        elif self.allocation_policy == SliceAllocationPolicy.PROPORTIONAL:
            # Select largest slices
            sorted_slices = sorted(
                available_slices,
                key=lambda s: s.compute_capacity,
                reverse=True
            )
            return sorted_slices[:count]
        
        else:  # ADAPTIVE
            # Use historical data to select best slices
            return self._adaptive_slice_selection(available_slices, count, expert_id)
    
    def _adaptive_slice_selection(
        self,
        available_slices: List[GPUSlice],
        count: int,
        expert_id: int
    ) -> List[GPUSlice]:
        """Adaptive slice selection based on historical performance."""
        # For now, use dynamic policy
        # In production, this would use ML to predict best allocation
        return self._select_slices(available_slices, count, expert_id)
    
    def _evict_low_priority_experts(
        self,
        required_slices: int,
        priority: int
    ) -> List[GPUSlice]:
        """Evict low-priority experts to free up slices."""
        # Find experts with lower priority
        candidates = [
            (eid, alloc) for eid, alloc in self.expert_allocations.items()
            if alloc.priority < priority
        ]
        
        # Sort by priority (lowest first)
        candidates.sort(key=lambda x: x[1].priority)
        
        freed_slices = []
        for expert_id, allocation in candidates:
            if len(freed_slices) >= required_slices:
                break
            
            # Release this expert's slices
            self.release_slices(expert_id)
            
            # Add to freed slices
            for slice_id in allocation.slice_ids:
                slice_obj = self.slices[slice_id]
                if slice_obj.assigned_expert is None:
                    freed_slices.append(slice_obj)
        
        self.reallocation_count += 1
        return freed_slices
    
    def release_slices(self, expert_id: int):
        """Release slices allocated to an expert."""
        if expert_id not in self.expert_allocations:
            return
        
        allocation = self.expert_allocations[expert_id]
        
        # Mark slices as free
        for slice_id in allocation.slice_ids:
            self.slices[slice_id].assigned_expert = None
            self.slices[slice_id].utilization = 0.0
        
        del self.expert_allocations[expert_id]
        logger.info(f"Released slices for expert {expert_id}")
    
    def update_slice_utilization(self, slice_id: int, utilization: float):
        """Update utilization metric for a slice."""
        if 0 <= slice_id < len(self.slices):
            self.slices[slice_id].utilization = utilization
            
            # Track history
            if slice_id not in self.slice_utilization_history:
                self.slice_utilization_history[slice_id] = []
            self.slice_utilization_history[slice_id].append(utilization)
    
    def get_allocation(self, expert_id: int) -> Optional[ExpertAllocation]:
        """Get allocation for an expert."""
        return self.expert_allocations.get(expert_id)
    
    def get_slice_info(self, slice_id: int) -> Optional[GPUSlice]:
        """Get information about a slice."""
        if 0 <= slice_id < len(self.slices):
            return self.slices[slice_id]
        return None
    
    def get_stats(self) -> Dict:
        """Get statistics about GPU slice usage."""
        total_allocated = sum(1 for s in self.slices if s.assigned_expert is not None)
        avg_utilization = sum(s.utilization for s in self.slices) / len(self.slices)
        
        return {
            'total_slices': self.total_slices,
            'allocated_slices': total_allocated,
            'free_slices': self.total_slices - total_allocated,
            'avg_utilization': avg_utilization,
            'num_experts': len(self.expert_allocations),
            'reallocation_count': self.reallocation_count,
            'mig_enabled': self.mig_manager.is_mig_enabled() if self.mig_manager else False
        }
    
    def optimize_allocations(self):
        """Optimize current allocations based on utilization history."""
        # Identify underutilized allocations
        for expert_id, allocation in list(self.expert_allocations.items()):
            avg_util = sum(
                self.slices[sid].utilization for sid in allocation.slice_ids
            ) / len(allocation.slice_ids)
            
            # If significantly underutilized, consider reducing allocation
            if avg_util < 0.3 and len(allocation.slice_ids) > 1:
                logger.info(f"Expert {expert_id} underutilized ({avg_util:.2f}), considering reallocation")
                # In production, implement reallocation logic here
