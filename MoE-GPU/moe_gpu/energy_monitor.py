"""
Energy and Performance Monitoring for Expert-Sliced GPU Scheduling.
Tracks power consumption, GPU utilization, and performance metrics.
"""

import torch
import pynvml
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnergyMetrics:
    """Energy and performance metrics for a time window."""
    timestamp: float
    power_watts: float
    gpu_utilization: float
    memory_utilization: float
    temperature_celsius: float
    sm_clock_mhz: float
    memory_clock_mhz: float
    throughput_tokens_per_sec: float = 0.0
    energy_per_token_joules: float = 0.0


@dataclass
class ExpertEnergyProfile:
    """Energy profile for a specific expert."""
    expert_id: int
    total_energy_joules: float = 0.0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    avg_power_watts: float = 0.0
    measurements: List[EnergyMetrics] = field(default_factory=list)
    
    def add_measurement(self, power: float, tokens: int, duration: float):
        """Add a measurement to the profile."""
        energy = power * duration
        self.total_energy_joules += energy
        self.total_tokens += tokens
        self.total_time_seconds += duration
        self.avg_power_watts = self.total_energy_joules / self.total_time_seconds if self.total_time_seconds > 0 else 0
    
    def get_energy_per_token(self) -> float:
        """Calculate energy per token."""
        return self.total_energy_joules / self.total_tokens if self.total_tokens > 0 else 0


class EnergyMonitor:
    """
    Monitors GPU energy consumption and performance metrics.
    Uses NVML to track real-time power and utilization.
    """
    
    def __init__(self, device_index: int = 0, sampling_interval: float = 0.1):
        self.device_index = device_index
        self.sampling_interval = sampling_interval
        self.device_handle = None
        self.monitoring_enabled = False
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=10000)
        self.expert_profiles: Dict[int, ExpertEnergyProfile] = {}
        
        # Timing
        self.start_time = None
        self.last_sample_time = None
        
        # Initialize NVML
        self._initialize_nvml()
    
    def _initialize_nvml(self):
        """Initialize NVML for GPU monitoring."""
        try:
            pynvml.nvmlInit()
            self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.monitoring_enabled = True
            logger.info(f"Energy monitoring enabled for GPU {self.device_index}")
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}. Energy monitoring disabled.")
            self.monitoring_enabled = False
    
    def start_monitoring(self):
        """Start energy monitoring."""
        self.start_time = time.time()
        self.last_sample_time = self.start_time
    
    def sample_metrics(self) -> Optional[EnergyMetrics]:
        """Sample current GPU metrics."""
        if not self.monitoring_enabled:
            return None
        
        try:
            current_time = time.time()
            
            # Power consumption (in milliwatts, convert to watts)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.device_handle)
            power_watts = power_mw / 1000.0
            
            # Utilization rates
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.device_handle)
            gpu_util = utilization.gpu
            memory_util = utilization.memory
            
            # Temperature
            temperature = pynvml.nvmlDeviceGetTemperature(
                self.device_handle,
                pynvml.NVML_TEMPERATURE_GPU
            )
            
            # Clock speeds
            sm_clock = pynvml.nvmlDeviceGetClockInfo(
                self.device_handle,
                pynvml.NVML_CLOCK_SM
            )
            memory_clock = pynvml.nvmlDeviceGetClockInfo(
                self.device_handle,
                pynvml.NVML_CLOCK_MEM
            )
            
            metrics = EnergyMetrics(
                timestamp=current_time,
                power_watts=power_watts,
                gpu_utilization=gpu_util,
                memory_utilization=memory_util,
                temperature_celsius=temperature,
                sm_clock_mhz=sm_clock,
                memory_clock_mhz=memory_clock
            )
            
            self.metrics_history.append(metrics)
            self.last_sample_time = current_time
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to sample metrics: {e}")
            return None
    
    def record_expert_execution(
        self,
        expert_id: int,
        num_tokens: int,
        start_time: float,
        end_time: float
    ):
        """
        Record energy consumption for an expert execution.
        
        Args:
            expert_id: ID of the expert
            num_tokens: Number of tokens processed
            start_time: Start timestamp
            end_time: End timestamp
        """
        if not self.monitoring_enabled:
            return
        
        duration = end_time - start_time
        
        # Get average power during execution
        relevant_metrics = [
            m for m in self.metrics_history
            if start_time <= m.timestamp <= end_time
        ]
        
        if relevant_metrics:
            avg_power = np.mean([m.power_watts for m in relevant_metrics])
        else:
            # Sample current power if no history
            current_metrics = self.sample_metrics()
            avg_power = current_metrics.power_watts if current_metrics else 0
        
        # Get or create expert profile
        if expert_id not in self.expert_profiles:
            self.expert_profiles[expert_id] = ExpertEnergyProfile(expert_id=expert_id)
        
        profile = self.expert_profiles[expert_id]
        profile.add_measurement(avg_power, num_tokens, duration)
    
    def get_expert_profile(self, expert_id: int) -> Optional[ExpertEnergyProfile]:
        """Get energy profile for an expert."""
        return self.expert_profiles.get(expert_id)
    
    def get_total_energy(self) -> float:
        """Calculate total energy consumed since monitoring started."""
        if not self.metrics_history:
            return 0.0
        
        total_energy = 0.0
        for i in range(1, len(self.metrics_history)):
            prev_metrics = self.metrics_history[i-1]
            curr_metrics = self.metrics_history[i]
            
            time_delta = curr_metrics.timestamp - prev_metrics.timestamp
            avg_power = (prev_metrics.power_watts + curr_metrics.power_watts) / 2
            
            total_energy += avg_power * time_delta
        
        return total_energy
    
    def get_average_power(self) -> float:
        """Calculate average power consumption."""
        if not self.metrics_history:
            return 0.0
        
        return np.mean([m.power_watts for m in self.metrics_history])
    
    def get_average_utilization(self) -> Dict[str, float]:
        """Calculate average GPU utilization."""
        if not self.metrics_history:
            return {'gpu': 0.0, 'memory': 0.0}
        
        return {
            'gpu': np.mean([m.gpu_utilization for m in self.metrics_history]),
            'memory': np.mean([m.memory_utilization for m in self.metrics_history])
        }
    
    def get_efficiency_metrics(self) -> Dict:
        """Calculate efficiency metrics across all experts."""
        if not self.expert_profiles:
            return {}
        
        total_tokens = sum(p.total_tokens for p in self.expert_profiles.values())
        total_energy = sum(p.total_energy_joules for p in self.expert_profiles.values())
        total_time = sum(p.total_time_seconds for p in self.expert_profiles.values())
        
        return {
            'total_tokens': total_tokens,
            'total_energy_joules': total_energy,
            'total_time_seconds': total_time,
            'tokens_per_joule': total_tokens / total_energy if total_energy > 0 else 0,
            'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
            'avg_power_watts': total_energy / total_time if total_time > 0 else 0
        }
    
    def get_expert_comparison(self) -> List[Dict]:
        """Compare energy efficiency across experts."""
        comparison = []
        
        for expert_id, profile in self.expert_profiles.items():
            comparison.append({
                'expert_id': expert_id,
                'total_tokens': profile.total_tokens,
                'energy_per_token': profile.get_energy_per_token(),
                'avg_power_watts': profile.avg_power_watts,
                'total_time_seconds': profile.total_time_seconds,
                'throughput': profile.total_tokens / profile.total_time_seconds if profile.total_time_seconds > 0 else 0
            })
        
        # Sort by energy efficiency (lower is better)
        comparison.sort(key=lambda x: x['energy_per_token'])
        
        return comparison
    
    def get_real_time_stats(self) -> Dict:
        """Get real-time statistics."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        
        return {
            'current_power_watts': latest.power_watts,
            'current_gpu_utilization': latest.gpu_utilization,
            'current_memory_utilization': latest.memory_utilization,
            'current_temperature': latest.temperature_celsius,
            'sm_clock_mhz': latest.sm_clock_mhz,
            'memory_clock_mhz': latest.memory_clock_mhz,
            'monitoring_duration': time.time() - self.start_time if self.start_time else 0
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to CSV file."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'timestamp', 'power_watts', 'gpu_utilization', 'memory_utilization',
                'temperature_celsius', 'sm_clock_mhz', 'memory_clock_mhz'
            ])
            
            # Write data
            for metrics in self.metrics_history:
                writer.writerow([
                    metrics.timestamp,
                    metrics.power_watts,
                    metrics.gpu_utilization,
                    metrics.memory_utilization,
                    metrics.temperature_celsius,
                    metrics.sm_clock_mhz,
                    metrics.memory_clock_mhz
                ])
        
        logger.info(f"Exported {len(self.metrics_history)} metrics to {filepath}")
    
    def reset(self):
        """Reset all metrics."""
        self.metrics_history.clear()
        self.expert_profiles.clear()
        self.start_time = None
        self.last_sample_time = None
    
    def __del__(self):
        """Cleanup NVML."""
        if self.monitoring_enabled:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class PerformanceComparator:
    """
    Compares performance between different MoE implementations.
    """
    
    def __init__(self):
        self.baseline_metrics: Dict = {}
        self.optimized_metrics: Dict = {}
    
    def record_baseline(self, metrics: Dict):
        """Record baseline implementation metrics."""
        self.baseline_metrics = metrics.copy()
    
    def record_optimized(self, metrics: Dict):
        """Record optimized implementation metrics."""
        self.optimized_metrics = metrics.copy()
    
    def compute_improvements(self) -> Dict:
        """Compute improvement percentages."""
        if not self.baseline_metrics or not self.optimized_metrics:
            return {}
        
        improvements = {}
        
        # Throughput improvement
        baseline_throughput = self.baseline_metrics.get('tokens_per_second', 0)
        optimized_throughput = self.optimized_metrics.get('tokens_per_second', 0)
        
        if baseline_throughput > 0:
            improvements['throughput_improvement'] = (
                (optimized_throughput - baseline_throughput) / baseline_throughput * 100
            )
        
        # Energy efficiency improvement
        baseline_energy = self.baseline_metrics.get('energy_per_token', 0)
        optimized_energy = self.optimized_metrics.get('energy_per_token', 0)
        
        if baseline_energy > 0:
            improvements['energy_efficiency_improvement'] = (
                (baseline_energy - optimized_energy) / baseline_energy * 100
            )
        
        # GPU utilization improvement
        baseline_util = self.baseline_metrics.get('avg_gpu_utilization', 0)
        optimized_util = self.optimized_metrics.get('avg_gpu_utilization', 0)
        
        if baseline_util > 0:
            improvements['utilization_improvement'] = (
                (optimized_util - baseline_util) / baseline_util * 100
            )
        
        return improvements
    
    def generate_comparison_report(self) -> str:
        """Generate a text report comparing implementations."""
        improvements = self.compute_improvements()
        
        report = "=" * 60 + "\n"
        report += "Performance Comparison Report\n"
        report += "=" * 60 + "\n\n"
        
        report += "Baseline Implementation:\n"
        report += f"  Throughput: {self.baseline_metrics.get('tokens_per_second', 0):.2f} tokens/sec\n"
        report += f"  Energy per token: {self.baseline_metrics.get('energy_per_token', 0):.6f} J\n"
        report += f"  GPU Utilization: {self.baseline_metrics.get('avg_gpu_utilization', 0):.2f}%\n\n"
        
        report += "Optimized Implementation (Expert-Sliced):\n"
        report += f"  Throughput: {self.optimized_metrics.get('tokens_per_second', 0):.2f} tokens/sec\n"
        report += f"  Energy per token: {self.optimized_metrics.get('energy_per_token', 0):.6f} J\n"
        report += f"  GPU Utilization: {self.optimized_metrics.get('avg_gpu_utilization', 0):.2f}%\n\n"
        
        report += "Improvements:\n"
        report += f"  Throughput: {improvements.get('throughput_improvement', 0):+.2f}%\n"
        report += f"  Energy Efficiency: {improvements.get('energy_efficiency_improvement', 0):+.2f}%\n"
        report += f"  GPU Utilization: {improvements.get('utilization_improvement', 0):+.2f}%\n"
        
        report += "=" * 60 + "\n"
        
        return report
