"""Test problem1_convergence function."""

import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hw6_cupy import problem1_convergence


def test_problem1_convergence():
    """Test problem1_convergence function."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        
        # Run convergence analysis
        print("Running problem1_convergence...")
        problem1_convergence(outdir)
        
        # Check output files were created
        expected_files = [
            "convergence_euler.csv",
            "convergence_rk2.csv",
            "convergence_rk4.csv",
            "problem1_error_vs_dt.png"
        ]
        
        for filename in expected_files:
            filepath = outdir / filename
            assert filepath.exists(), f"Expected file not created: {filename}"
            print(f"✓ {filename} created")
        
        # Check CSV files contain data
        for csv_file in ["convergence_euler.csv", "convergence_rk2.csv", "convergence_rk4.csv"]:
            filepath = outdir / csv_file
            content = filepath.read_text()
            assert "dt" in content, f"{csv_file} missing 'dt' header"
            assert "error" in content, f"{csv_file} missing 'error' header"
            lines = content.strip().split('\n')
            # At least header + some data rows
            assert len(lines) > 2, f"{csv_file} has insufficient data"
            print(f"✓ {csv_file} contains convergence data")
        
        # Check plot file is not empty
        plot_file = outdir / "problem1_error_vs_dt.png"
        assert plot_file.stat().st_size > 0, "Plot file is empty"
        print(f"✓ problem1_error_vs_dt.png contains plot data")
        
        print("\n✓ problem1_convergence test passed!")


if __name__ == "__main__":
    test_problem1_convergence()
