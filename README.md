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

## Notes

- Some directories include generated artifacts (for example, `__pycache__`, binaries, and result files).
- Assignment-specific build and run instructions are typically stored within each homework folder.
