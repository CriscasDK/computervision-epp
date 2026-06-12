from datetime import datetime
import psutil
import logging
import statistics
import json
import os
import sys
import pytz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

class PerformanceMonitor:
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.vram_usage = []
        self.fps_values = []
        self.start_time = datetime.now(pytz.timezone("America/Bogota"))
        self.frames_processed = 0

        self.gpu_handle = None
        if GPU_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.gpu_handle = None

    def update(self, fps):
        """Llamado en cada frame procesado."""
        self.frames_processed += 1
        if fps:
            self.fps_values.append(fps)
        self.cpu_usage.append(psutil.cpu_percent(interval=None))
        self.memory_usage.append(psutil.virtual_memory().percent)

        if self.gpu_handle:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.gpu_usage.append(gpu_util.gpu)
                self.vram_usage.append(mem_info.used / (1024 ** 2))  # en MiB
            except Exception:
                pass

    def finalize(self, output_dir="logs"):
        """Guarda un archivo JSON con el resumen de métricas."""
        duration = (datetime.now(pytz.timezone("America/Bogota")) - self.start_time).total_seconds()
        summary = {
            "timestamp": self.start_time.isoformat(),
            "duration_sec": round(duration, 2),
            "frames_processed": self.frames_processed,
            "fps_mean": round(statistics.mean(self.fps_values), 2) if self.fps_values else 0,
            "fps_max": round(max(self.fps_values), 2) if self.fps_values else 0,
            "cpu_mean_percent": round(statistics.mean(self.cpu_usage), 2) if self.cpu_usage else 0,
            "memory_mean_percent": round(statistics.mean(self.memory_usage), 2) if self.memory_usage else 0,
            "gpu_mean_percent": round(statistics.mean(self.gpu_usage), 2) if self.gpu_usage else None,
            "vram_mean_mib": round(statistics.mean(self.vram_usage), 2) if self.vram_usage else None,
        }

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = f"{output_dir}/performance_metrics_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        logger.info(f"📊 Métricas de rendimiento guardadas en {output_path}")
        return summary