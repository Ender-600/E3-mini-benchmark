"""Logging utilities for E³ Mini-Benchmark."""

import logging
import subprocess
import threading
import time
from typing import Dict, Any, Optional
import torch
import psutil


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


class PowerMonitor:
    """Background thread to monitor GPU power usage."""
    
    def __init__(self):
        self.power_readings = []
        self.monitoring = False
        self.thread = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.used_cpu_estimate = False
        self._final_stats: Optional[Dict[str, float]] = None
        
    def start(self):
        """Start power monitoring."""
        # Reset state to support re-use across experiments
        self.power_readings = []
        self.monitoring = True
        self.used_cpu_estimate = False
        self._final_stats = None
        self.start_time = time.time()
        self.end_time = None
        self.thread = threading.Thread(target=self._monitor_power, daemon=True)
        self.thread.start()
        
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return power statistics."""
        if self._final_stats is not None:
            # idempotent stop: return cached statistics
            return self._final_stats
        
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        self.end_time = time.time()
        duration_seconds = 0.0
        if self.start_time is not None and self.end_time is not None:
            duration_seconds = max(self.end_time - self.start_time, 0.0)
        
        if not self.power_readings:
            stats = {"avg_watt": None, "kwh": None, "duration_seconds": duration_seconds}
            self._final_stats = stats
            return stats
        
        avg_watt = sum(self.power_readings) / len(self.power_readings)
        
        if duration_seconds <= 0.0:
            estimated_duration = len(self.power_readings)
            duration_seconds = estimated_duration if estimated_duration > 0 else 0.0
        
        kwh = None
        if duration_seconds > 0.0:
            # Convert average watts to kilowatt-hours using actual elapsed time
            kwh = (avg_watt / 1000.0) * (duration_seconds / 3600.0)
        
        # If we had to fall back to CPU-based estimation, mark readings as unreliable
        if self.used_cpu_estimate:
            avg_watt = None
            kwh = None
        
        stats = {"avg_watt": avg_watt, "kwh": kwh, "duration_seconds": duration_seconds}
        self._final_stats = stats
        return stats
        
    def _monitor_power(self):
        """Monitor GPU power in background thread."""
        while self.monitoring:
            try:
                # Try nvidia-smi first
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1.0
                )
                if result.returncode == 0:
                    # Handle multiple GPUs: nvidia-smi returns one value per line
                    # Sum power across all GPUs to get total system power consumption
                    lines = result.stdout.strip().split('\n')
                    if lines and lines[0]:  # Check if we have at least one non-empty line
                        try:
                            # Parse all GPU power values
                            power_values = [float(line.strip()) for line in lines if line.strip()]
                            if power_values:
                                # Sum power across all GPUs for total system power consumption
                                total_power = sum(power_values)
                                if total_power > 0:  # Valid power reading
                                    self.power_readings.append(total_power)
                            else:
                                # Fallback to CPU power estimation
                                cpu_percent = psutil.cpu_percent(interval=0.1)
                                estimated_power = cpu_percent * 0.1  # Rough estimate
                                self.power_readings.append(estimated_power)
                                self.used_cpu_estimate = True
                        except (ValueError, IndexError):
                            # If parsing fails, fallback to CPU estimation
                            cpu_percent = psutil.cpu_percent(interval=0.1)
                            estimated_power = cpu_percent * 0.1
                            self.power_readings.append(estimated_power)
                            self.used_cpu_estimate = True
                    else:
                        # Empty output, fallback to CPU estimation
                        cpu_percent = psutil.cpu_percent(interval=0.1)
                        estimated_power = cpu_percent * 0.1
                        self.power_readings.append(estimated_power)
                        self.used_cpu_estimate = True
                else:
                    # nvidia-smi failed, fallback to CPU power estimation
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    estimated_power = cpu_percent * 0.1  # Rough estimate
                    self.power_readings.append(estimated_power)
                    self.used_cpu_estimate = True
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                # Fallback to CPU estimation
                cpu_percent = psutil.cpu_percent(interval=0.1)
                estimated_power = cpu_percent * 0.1
                self.power_readings.append(estimated_power)
                self.used_cpu_estimate = True
            except Exception:
                pass  # Ignore other errors
                
            time.sleep(1.0)


def log_experiment_info(
    exp_id: str,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    start_time: float,
    end_time: float,
    power_monitor: Optional[PowerMonitor] = None
) -> Dict[str, Any]:
    """Log comprehensive experiment information."""
    
    # Get GPU information
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "max_memory_gb": torch.cuda.max_memory_allocated(0) / 1024**3,
            "cuda_version": torch.version.cuda
        }
    
    # Get power statistics
    power_stats = {}
    if power_monitor:
        power_stats = power_monitor.stop()
    
    # Get git commit if available
    commit_hash = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5.0
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()[:8]
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    experiment_log = {
        "exp_id": exp_id,
        "commit": commit_hash,
        "config": config,
        "metrics": metrics,
        "timing": {
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": end_time - start_time
        },
        "resources": gpu_info,
        "power": power_stats,
        "timestamp": time.time()
    }
    
    return experiment_log
