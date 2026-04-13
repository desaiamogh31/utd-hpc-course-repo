"""Test problem1_validate function."""

import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hw6_cupy import problem1_validate


def test_problem1_validate():
    """Test problem1_validate function."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        
        # Run validation
        print("Running problem1_validate...")
        problem1_validate(outdir)
        
        # Check output file was created
        validation_file = outdir / "validation.txt"
        assert validation_file.exists(), f"Validation file not created: {validation_file}"
        
        # Read and print results
        results = validation_file.read_text()
        print("\n=== Validation Results ===")
        print(results)
        print("=" * 50)
        
        # Check that results contain expected methods
        assert "euler" in results.lower(), "Euler method not in results"
        assert "rk2" in results.lower() or "rk4" in results.lower(), "RK methods not in results"
        
        print("✓ problem1_validate test passed!")


if __name__ == "__main__":
    test_problem1_validate()
