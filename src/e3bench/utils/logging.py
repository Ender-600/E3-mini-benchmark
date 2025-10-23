"""Logging utilities for E³ Mini-Benchmark."""

import logging
import subprocess
import threading
import time
from typing import Dict, Any, Optional
import torch
import psutil
import os


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
        
    def start(self):
        """Start power monitoring."""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_power, daemon=True)
        self.thread.start()
        
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return power statistics."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
        if not self.power_readings:
            return {"avg_watt": 0.0, "kwh": 0.0}
            
        avg_watt = sum(self.power_readings) / len(self.power_readings)
        # Rough kWh estimate (not precise without time tracking)
        kwh = avg_watt * len(self.power_readings) / 3600.0 / 1000.0
        
        return {"avg_watt": avg_watt, "kwh": kwh}
        
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
                    power = float(result.stdout.strip())
                    if power > 0:  # Valid power reading
                        self.power_readings.append(power)
                else:
                    # Fallback to CPU power estimation
                    cpu_percent = psutil.cpu_percent()
                    estimated_power = cpu_percent * 0.1  # Rough estimate
                    self.power_readings.append(estimated_power)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
                # Fallback to CPU estimation
                cpu_percent = psutil.cpu_percent()
                estimated_power = cpu_percent * 0.1
                self.power_readings.append(estimated_power)
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
