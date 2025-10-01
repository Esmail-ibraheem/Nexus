"""
CUDA Graph Manager for Expert-Sliced GPU Scheduling.
Captures and replays expert computations using CUDA graphs for reduced kernel launch overhead.
"""

import torch
import torch.cuda as cuda
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CUDAGraphManager:
    """
    Manages CUDA graphs for expert computations.
    Captures frequently used expert execution patterns and replays them efficiently.
    """
    
    def __init__(self, enable_graphs: bool = True):
        self.enable_graphs = enable_graphs and torch.cuda.is_available()
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[int, torch.Tensor] = {}
        self.static_outputs: Dict[int, torch.Tensor] = {}
        self.capture_count: Dict[int, int] = {}
        self.warmup_steps = 3
        
    def should_capture(self, expert_id: int, min_calls: int = 10) -> bool:
        """Determine if we should capture a graph for this expert."""
        if not self.enable_graphs:
            return False
        
        count = self.capture_count.get(expert_id, 0)
        self.capture_count[expert_id] = count + 1
        
        # Capture after warmup and if called frequently
        return count >= self.warmup_steps and expert_id not in self.graphs
    
    def capture_expert_forward(
        self,
        expert_id: int,
        expert_module: torch.nn.Module,
        input_shape: Tuple[int, ...],
        device: torch.device
    ) -> bool:
        """
        Capture a CUDA graph for an expert's forward pass.
        
        Args:
            expert_id: ID of the expert
            expert_module: The expert module to capture
            input_shape: Shape of input tensor
            device: CUDA device
            
        Returns:
            True if capture was successful
        """
        if not self.enable_graphs:
            return False
        
        try:
            # Create static input/output tensors
            static_input = torch.randn(input_shape, device=device)
            
            # Warmup
            for _ in range(self.warmup_steps):
                _ = expert_module(static_input)
            
            # Capture graph
            graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(graph):
                static_output = expert_module(static_input)
            
            # Store graph and tensors
            self.graphs[expert_id] = graph
            self.static_inputs[expert_id] = static_input
            self.static_outputs[expert_id] = static_output
            
            logger.info(f"Captured CUDA graph for expert {expert_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to capture CUDA graph for expert {expert_id}: {e}")
            return False
    
    def replay_expert_forward(
        self,
        expert_id: int,
        input_tensor: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Replay a captured CUDA graph for an expert.
        
        Args:
            expert_id: ID of the expert
            input_tensor: Input tensor (must match captured shape)
            
        Returns:
            Output tensor if graph exists, None otherwise
        """
        if expert_id not in self.graphs:
            return None
        
        # Copy input to static buffer
        self.static_inputs[expert_id].copy_(input_tensor)
        
        # Replay graph
        self.graphs[expert_id].replay()
        
        # Return output (clone to avoid overwriting)
        return self.static_outputs[expert_id].clone()
    
    def clear_graph(self, expert_id: int):
        """Clear a captured graph for an expert."""
        if expert_id in self.graphs:
            del self.graphs[expert_id]
            del self.static_inputs[expert_id]
            del self.static_outputs[expert_id]
            logger.info(f"Cleared CUDA graph for expert {expert_id}")
    
    def clear_all_graphs(self):
        """Clear all captured graphs."""
        self.graphs.clear()
        self.static_inputs.clear()
        self.static_outputs.clear()
        logger.info("Cleared all CUDA graphs")
    
    def get_stats(self) -> Dict:
        """Get statistics about captured graphs."""
        return {
            'num_graphs': len(self.graphs),
            'expert_ids': list(self.graphs.keys()),
            'capture_counts': self.capture_count.copy()
        }


class StreamManager:
    """
    Manages CUDA streams for parallel expert execution.
    Assigns experts to different streams based on their GPU slice allocation.
    """
    
    def __init__(self, num_streams: int = 8):
        self.num_streams = num_streams
        self.streams: List[torch.cuda.Stream] = []
        self.expert_to_stream: Dict[int, int] = {}
        self.stream_usage: Dict[int, int] = {i: 0 for i in range(num_streams)}
        
        if torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
    
    def assign_expert_to_stream(self, expert_id: int, slice_id: int) -> int:
        """
        Assign an expert to a CUDA stream based on its GPU slice.
        
        Args:
            expert_id: ID of the expert
            slice_id: GPU slice ID assigned to this expert
            
        Returns:
            Stream index
        """
        # Map slice to stream (can be many-to-one)
        stream_idx = slice_id % self.num_streams
        self.expert_to_stream[expert_id] = stream_idx
        self.stream_usage[stream_idx] += 1
        return stream_idx
    
    def get_stream(self, expert_id: int) -> Optional[torch.cuda.Stream]:
        """Get the CUDA stream assigned to an expert."""
        if expert_id not in self.expert_to_stream:
            return None
        
        stream_idx = self.expert_to_stream[expert_id]
        return self.streams[stream_idx] if stream_idx < len(self.streams) else None
    
    def get_stream_by_index(self, stream_idx: int) -> Optional[torch.cuda.Stream]:
        """Get a CUDA stream by index."""
        if 0 <= stream_idx < len(self.streams):
            return self.streams[stream_idx]
        return None
    
    def synchronize_all(self):
        """Synchronize all streams."""
        for stream in self.streams:
            stream.synchronize()
    
    def synchronize_expert(self, expert_id: int):
        """Synchronize the stream assigned to an expert."""
        stream = self.get_stream(expert_id)
        if stream:
            stream.synchronize()
    
    def get_stats(self) -> Dict:
        """Get statistics about stream usage."""
        return {
            'num_streams': self.num_streams,
            'stream_usage': self.stream_usage.copy(),
            'expert_assignments': len(self.expert_to_stream)
        }
    
    def reset(self):
        """Reset stream assignments."""
        self.expert_to_stream.clear()
        self.stream_usage = {i: 0 for i in range(self.num_streams)}


class BatchScheduler:
    """
    Schedules expert batches to maximize GPU utilization.
    Groups experts with similar workloads and schedules them on appropriate streams.
    """
    
    def __init__(
        self,
        stream_manager: StreamManager,
        graph_manager: CUDAGraphManager
    ):
        self.stream_manager = stream_manager
        self.graph_manager = graph_manager
        self.batch_history: List[Dict] = []
    
    def schedule_expert_batch(
        self,
        expert_id: int,
        expert_module: torch.nn.Module,
        input_tensor: torch.Tensor,
        slice_id: int,
        use_graph: bool = True
    ) -> torch.Tensor:
        """
        Schedule an expert computation on the appropriate stream.
        
        Args:
            expert_id: ID of the expert
            expert_module: The expert module
            input_tensor: Input tensor
            slice_id: GPU slice ID
            use_graph: Whether to use CUDA graphs
            
        Returns:
            Output tensor
        """
        # Assign to stream
        stream_idx = self.stream_manager.assign_expert_to_stream(expert_id, slice_id)
        stream = self.stream_manager.get_stream_by_index(stream_idx)
        
        if stream is None:
            # Fallback to default stream
            return self._execute_expert(expert_id, expert_module, input_tensor, use_graph)
        
        # Execute on assigned stream
        with torch.cuda.stream(stream):
            output = self._execute_expert(expert_id, expert_module, input_tensor, use_graph)
        
        # Record batch info
        self.batch_history.append({
            'expert_id': expert_id,
            'stream_idx': stream_idx,
            'slice_id': slice_id,
            'batch_size': input_tensor.shape[0],
            'used_graph': use_graph and expert_id in self.graph_manager.graphs
        })
        
        return output
    
    def _execute_expert(
        self,
        expert_id: int,
        expert_module: torch.nn.Module,
        input_tensor: torch.Tensor,
        use_graph: bool
    ) -> torch.Tensor:
        """Execute expert computation, optionally using CUDA graphs."""
        
        # Try to use CUDA graph if available
        if use_graph and expert_id in self.graph_manager.graphs:
            output = self.graph_manager.replay_expert_forward(expert_id, input_tensor)
            if output is not None:
                return output
        
        # Check if we should capture a new graph
        if use_graph and self.graph_manager.should_capture(expert_id):
            self.graph_manager.capture_expert_forward(
                expert_id,
                expert_module,
                input_tensor.shape,
                input_tensor.device
            )
        
        # Regular execution
        return expert_module(input_tensor)
    
    def synchronize_and_collect(self) -> List[torch.Tensor]:
        """
        Synchronize all streams and collect results.
        
        Returns:
            List of output tensors
        """
        self.stream_manager.synchronize_all()
        return []
    
    def get_batch_stats(self) -> Dict:
        """Get statistics about batch scheduling."""
        if not self.batch_history:
            return {}
        
        total_batches = len(self.batch_history)
        graph_usage = sum(1 for b in self.batch_history if b['used_graph'])
        
        return {
            'total_batches': total_batches,
            'graph_usage_rate': graph_usage / total_batches if total_batches > 0 else 0,
            'avg_batch_size': sum(b['batch_size'] for b in self.batch_history) / total_batches,
            'stream_distribution': self._get_stream_distribution()
        }
    
    def _get_stream_distribution(self) -> Dict[int, int]:
        """Get distribution of batches across streams."""
        distribution = {}
        for batch in self.batch_history:
            stream_idx = batch['stream_idx']
            distribution[stream_idx] = distribution.get(stream_idx, 0) + 1
        return distribution
    
    def reset_history(self):
        """Reset batch history."""
        self.batch_history.clear()
