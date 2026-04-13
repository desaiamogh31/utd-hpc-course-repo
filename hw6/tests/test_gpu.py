#!/usr/bin/env python3
"""Simple test to verify CuPy and GPU availability."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend import get_backend, CUPY_AVAILABLE, sync, to_cpu


def test_cupy_available():
    """Check if CuPy is available."""
    print(f"CuPy available: {CUPY_AVAILABLE}")
    assert CUPY_AVAILABLE, "CuPy not installed or CUDA not available"


def test_gpu_backend():
    """Test getting GPU backend."""
    backend = get_backend(prefer_gpu=True)
    print(f"Backend: {backend.name}")
    print(f"Has GPU: {backend.has_gpu}")
    assert backend.has_gpu, "GPU backend not available"


def test_simple_operation():
    """Test a simple GPU operation."""
    backend = get_backend(prefer_gpu=True)
    xp = backend.xp
    
    # Create arrays on GPU
    a = xp.array([1.0, 2.0, 3.0])
    b = xp.array([4.0, 5.0, 6.0])
    
    # Perform operation
    c = a + b
    
    # Sync GPU
    sync(backend)
    
    # Convert back to NumPy for verification
    result = to_cpu(c)
    
    print(f"Result: {result}")
    expected = [5.0, 7.0, 9.0]
    assert all(abs(result[i] - expected[i]) < 1e-6 for i in range(len(result)))
    print("✓ Simple operation test passed")


if __name__ == "__main__":
    test_cupy_available()
    test_gpu_backend()
    test_simple_operation()
    print("\n✓ All tests passed!")
