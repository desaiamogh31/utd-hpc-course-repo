# HPC Course Repository

High-Performance Computing coursework repository. Each assignment has a separate directory, such as `hw1/`, `hw2/`, `hw3/`, etc.

## Repository Layout

### `hw1/`
Introductory hello-world style programs in multiple languages.

- `hello.cpp` - C++ hello world
- `hello.py` - Python hello world
- `hello.sh` - Shell script hello world
- `hello` - Compiled binary output
- `sample-github-copilot/` and `sample-github-copilot.py` - Copilot sample files

### `hw2/`
Quantum eigenvalue solver assignment.

- `src/eigen.py` - Main solver implementation
- `src/array_eigen.slurm` - SLURM job-array submission script
- `src/analysis.ipynb` - Analysis notebook
- `tests/` - Test suite

### `hw3/`
Monte Carlo estimation of pi using multiple approaches.

- `src/pi_python.py` - Pure Python loop
- `src/pi_numpy.py` - NumPy vectorized implementation
- `src/pi_numba.py` - Numba-accelerated implementation
- `src/main.cpp` plus generated binaries (`calc_pi`, `mc_pi`) - C++ implementation
- `results/` - Output/performance results

### `hw4/`
Parallel Lorentzian computation and scaling analysis using multiple frameworks.

Key implementations in `src/` include:

- `lorentzian.py` - Serial baseline
- `mp_lorentz.py` - `multiprocessing`
- `thread_lorentz.py` - `threading`
- `joblib_lorentz.py` - Joblib backend
- `mpire_lorentz.py` - MPIRE backend
- `ppe_lorentz.py` - `ProcessPoolExecutor`
- `numba_lorentz.py` - Numba
- `dask_lorentz.py` - Dask
- `async_lorentz.py` - asyncio-based variant
- `mpi_lorentz.py` - MPI (`mpi4py`)
- `mpi_lorentz_slurm.slurm` - SLURM launch script
- `lorentz.ipynb` - Analysis notebook

### `hw5/`
Hybrid MPI + OpenMP assignment, including Monte Carlo and N-body examples.

- `mc_mpi_omp.cc` - Monte Carlo pi with MPI + OpenMP
- `mpi_omp_atomic.cc` - OpenMP atomic-based reduction variant
- `mpi_omp_critical.cc` - OpenMP critical-section reduction variant
- `NBODY.cc` - N-body simulation code
- `nbody_serial.py` - Serial Python N-body reference
- `run_mpi_omp.txt` - Script to compile/run MPI+OpenMP variants

### `hw6/`
GPU-accelerated ODE solver implementation using CuPy and CUDA.

**Core Implementation:**
- `src/hw6_cupy.py` - Main CuPy implementation with 4 problems:
  - **Problem 1**: Explicit IVP solvers (Euler, RK2, RK4) - includes validation, convergence analysis, and timing benchmarks
  - **Problem 2**: Stability regions analysis in the complex plane
  - **Problem 3**: Batched logistic RK4 with GPU benchmarking
  - **Problem 4**: Stiff diagonal systems with implicit methods (trapezoidal, TR-BDF2) for stability analysis
- `src/backend.py` - Backend utilities:
  - `Backend` dataclass for NumPy/CuPy abstraction
  - `get_backend()` - Selects CPU or GPU backend
  - `Timer` - Context manager with GPU synchronization for accurate timing
  - `to_cpu()` - Converts GPU/CPU arrays to NumPy
  - `sync()` - GPU stream synchronization

**Testing & Benchmarking:**
- `tests/test_gpu.py` - Basic GPU availability and functionality test
- `tests/test_problem1_validate.py` - Validates CPU vs GPU results agreement
- `tests/test_problem1_convergence.py` - Tests convergence analysis outputs
- `benchmarks/` - Performance benchmark results

**Running on HPC:**
```bash
# Request GPU on CMT partition (has NVIDIA A30 GPUs)
#SBATCH --partition=cmt
#SBATCH --gpus=1

# Load modules
module load cuda
module load python

# Activate conda environment
conda activate hpc-s26

# Install CuPy if needed
pip install cupy

# Run tests
python tests/test_gpu.py
python tests/test_problem1_convergence.py

# Run specific problem
python src/hw6_cupy.py problem1 --outdir results/p1
python src/hw6_cupy.py all --outdir results/
```

## Notes

- Some directories include generated artifacts (for example, `__pycache__`, binaries, and result files).
- Assignment-specific build and run instructions are typically stored within each homework folder.
