"""Hardware-agnostic PyTorch device dispatch and accelerator management.

Automatically detects and configures:
  1. NVIDIA CUDA (with cuDNN auto-tuning and VRAM monitoring)
  2. Apple Silicon MPS (Metal Performance Shaders on macOS)
  3. CPU (with multi-core thread optimization)
"""

import os
import platform
from typing import Any, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


class DeviceManager:
    """Manages tensor device selection, hardware telemetry, and memory flushing."""

    @staticmethod
    def is_torch_available() -> bool:
        """Return True if PyTorch is installed and importable."""
        return TORCH_AVAILABLE

    @classmethod
    def get_optimal_device(cls, preferred: Optional[str] = "auto") -> "torch.device":
        """Determine and return the most performant compute device available.

        Args:
            preferred: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns:
            torch.device instance configured for the selected hardware.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed in the active Python environment. "
                "Please run `pip install torch` to enable deep learning modules."
            )

        pref = (preferred or "auto").lower()

        # Explicit preference override
        if pref == "cuda":
            if torch.cuda.is_available():
                cls._configure_cuda()
                return torch.device("cuda")
            raise RuntimeError("CUDA was explicitly requested, but no CUDA-enabled GPU was detected.")

        if pref == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            raise RuntimeError("Apple Silicon MPS was requested, but MPS is not supported or available.")

        if pref == "cpu":
            return cls._configure_cpu()

        # Automatic detection hierarchy: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            cls._configure_cuda()
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Check if MPS backend was built with the current PyTorch binary
            if hasattr(torch.backends.mps, "is_built") and torch.backends.mps.is_built():
                return torch.device("mps")

        return cls._configure_cpu()

    @staticmethod
    def _configure_cuda() -> None:
        """Enable cuDNN performance flags for NVIDIA architectures."""
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            # cuDNN autotuner searches for fastest convolution/GEMM algorithms
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def _configure_cpu() -> "torch.device":
        """Tune CPU thread pools for parallel tensor operations."""
        cpu_count = os.cpu_count() or 4
        # Allocate half of physical cores to avoid saturating host OS
        optimal_threads = max(1, min(cpu_count, 8))
        try:
            torch.set_num_threads(optimal_threads)
        except RuntimeError:
            pass  # Already initialized
        return torch.device("cpu")

    @classmethod
    def get_device_info(cls, device: Optional["torch.device"] = None) -> dict[str, Any]:
        """Collect diagnostic telemetry for the active device and system hardware."""
        if not TORCH_AVAILABLE:
            return {
                "torch_available": False,
                "os": platform.system(),
                "machine": platform.machine(),
            }

        dev = device or cls.get_optimal_device()
        dev_type = dev.type

        info: dict[str, Any] = {
            "torch_available": True,
            "torch_version": torch.__version__,
            "selected_device": str(dev),
            "device_type": dev_type,
            "os": platform.system(),
            "machine": platform.machine(),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }

        # NVIDIA GPU Diagnostics
        if dev_type == "cuda" and torch.cuda.is_available():
            current_idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(current_idx)
            info.update({
                "gpu_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_mb": round(props.total_memory / (1024**2), 2),
                "allocated_memory_mb": round(torch.cuda.memory_allocated(current_idx) / (1024**2), 2),
                "cached_memory_mb": round(torch.cuda.memory_reserved(current_idx) / (1024**2), 2),
            })

        # Apple Silicon MPS Diagnostics
        elif dev_type == "mps":
            info["gpu_name"] = f"Apple Silicon GPU ({platform.processor() or platform.machine()})"

        # CPU Diagnostics
        elif dev_type == "cpu":
            info["cpu_threads"] = torch.get_num_threads()

        return info

    @classmethod
    def empty_cache(cls, device: Optional["torch.device"] = None) -> None:
        """Release unoccupied cached memory back to the operating system."""
        if not TORCH_AVAILABLE:
            return

        dev = device or cls.get_optimal_device()
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


# Convenience global accessor
def get_device(preferred: Optional[str] = "auto") -> "torch.device":
    """Retrieve the optimal available compute device."""
    return DeviceManager.get_optimal_device(preferred=preferred)
