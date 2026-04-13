from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False


@dataclass(frozen=True)
class Backend:
    xp: Any
    name: str
    has_gpu: bool
    device_id: Optional[int] = None


def get_backend(prefer_gpu: bool = True, device_id: int = 0) -> Backend:
    """Return a working NumPy or CuPy backend."""
    if prefer_gpu and CUPY_AVAILABLE:
        try:
            dev = cp.cuda.Device(device_id)
            dev.use()
            _ = cp.zeros((1,), dtype=cp.float32)
            return Backend(xp=cp, name="cupy", has_gpu=True, device_id=device_id)
        except Exception:
            pass
    return Backend(xp=np, name="numpy", has_gpu=False, device_id=None)


def to_cpu(x: Any) -> np.ndarray:
    """Convert backend array/scalar to NumPy on host."""
    if CUPY_AVAILABLE and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def sync(backend: Backend) -> None:
    """Synchronize GPU stream for accurate timing."""
    if backend.has_gpu:
        cp.cuda.Stream.null.synchronize()


class Timer:
    """Wall timer that synchronizes GPU before/after timing."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.dt = None

    def __enter__(self):
        import time
        sync(self.backend) # Ensure sync before starting timer
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        import time
        sync(self.backend) # Ensure sync before stopping timer
        self.dt = time.perf_counter() - self._t0
        return False
