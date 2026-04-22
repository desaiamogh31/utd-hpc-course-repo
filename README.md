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
  - **Problem 1**: Explicit IVP solvers (Euler, RK2, RK4) - Convergence analysis and timing benchmarks
  - **Problem 2**: Stability regions analysis in the complex plane
  - **Problem 3**: Batched logistic RK4 with GPU scaling analysis
    - GPU wins at N ≥ 50,000 trajectories (17× speedup at N=10⁶)
    - Demonstrates kernel launch and memory transfer overhead amortization
  - **Problem 4**: Stiff diagonal systems with implicit methods (Trapezoidal, TR-BDF2)
    - Demonstrates explicit Euler instability on stiff systems
    - Compares A-stable vs L-stable methods
    - **Updated**: Subplot visualizations for clarity (damping & stiffness demo)

- `src/backend.py` - Backend utilities:
  - `Backend` dataclass for NumPy/CuPy abstraction
  - `get_backend()` - Selects CPU or GPU backend
  - `Timer` - Context manager with GPU synchronization
  - `to_cpu()` - Converts arrays to NumPy
  - `sync()` - GPU stream synchronization

**SLURM Job Scripts:**
- `src/run_problem1.sh` - Convergence and timing analysis
- `src/run_problem2.sh` - Complex dynamics in frequency domain
- `src/run_problem3.sh` - Batched GPU computing (N=1K to 1M)
- `src/run_problem4.sh` - Stiff system analysis with implicit methods

**Key Results:**
| Problem | GPU Speedup | Notes |
|---------|------------|-------|
| P1 | ~2-4× | Memory-bound, good GPU fit |
| P2 | - | Stability analysis (not benchmarked) |
| P3 @ N=10⁶ | **17×** | Embarrassingly parallel, compute-bound |
| P3 @ N=100K | **3.4×** | Break-even overhead amortization |
| P3 @ N=1K | 0.04× | Overhead dominates, underutilized |
| P4 | 0.04× | Light computation, memory transfer overhead |

**Running on HPC:**
```bash
# Submit SLURM jobs
sbatch src/run_problem1.sh    # Convergence
sbatch src/run_problem2.sh    # Stability regions
sbatch src/run_problem3.sh    # GPU scaling (shows 17× speedup)
sbatch src/run_problem4.sh    # Stiffness analysis

# Or run directly
python src/hw6_cupy.py problem1 --outdir results/problem1
python src/hw6_cupy.py problem3 --Ns 1000 10000 100000 1000000
python src/hw6_cupy.py problem4 --d 1000000

# Monitor job
JOB_ID=<job_id>
scontrol show job $JOB_ID | grep -E "ElapsedTime|TimeLimit"
```

**Key Insights:**
- GPU overhead (kernel launch, memory transfer) amortizes only for large batch sizes
- Problem 3 demonstrates ideal GPU use case: N independent trajectories in parallel
- Problem 4 shows GPU limitations: small compute-to-memory ratio → CPU faster
- Implicit methods (TRBDF2) essential for stiff systems (explicit Euler diverges)
- Visualization improvements: subplot format for clarity (P4 damping & stiffness plots)

## Notes

- Some directories include generated artifacts (for example, `__pycache__`, binaries, and result files).
- Assignment-specific build and run instructions are typically stored within each homework folder.
