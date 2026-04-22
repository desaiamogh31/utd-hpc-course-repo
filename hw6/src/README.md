# HW6: GPU Computing with CuPy

High-performance computing assignment benchmarking CPU vs GPU implementations of ODE solvers and stiff system analysis using NumPy and CuPy.

## Problems Overview

### Problem 1: Explicit IVP Solvers
Validates and benchmarks three explicit Runge-Kutta methods (Euler, RK2, RK4) on exponential decay ODE:
- Computes convergence rates
- Measures timing performance
- Compares CPU vs GPU speedup
- **Output**: Convergence analysis and timing comparisons

### Problem 2: Complex Dynamics in Frequency Domain
Analyzes the complex growth factor for a semi-implicit method in frequency domain:
- Studies negative real extent
- Visualizes stability regions
- Benchmarks matrix operations
- **Output**: Stability region plots and frequency domain analysis

### Problem 3: Batched GPU Computing ⭐
Solves N independent logistic growth trajectories using RK4:
- **Key finding**: GPU wins when N ≥ 50,000 trajectories
- Demonstrates kernel launch and memory transfer overhead amortization
- Shows GPU parallelism benefits vs. Python loop overhead
- **Output**: Speedup and throughput curves

**Latest Changes:**
- Problem 3 demonstrates clear GPU scaling curves
- CPU bottleneck is the sequential Python time-stepping loop
- GPU shows 17× speedup at N=10⁶

### Problem 4: Stiff Systems & Implicit Methods (Updated)
Compares explicit and implicit methods for stiff diagonal systems:
- Demonstrates explicit Euler instability
- Compares A-stable (TR) vs L-stable (TRBDF2) methods
- Benchmarks implicit solver performance

**Latest Changes (Visualization Fixes):**
- **Subplot fix**: Split 6-line damping plot into 3 side-by-side subplots (one per α value)
  - Left: α=1 (non-stiff) — methods behave similarly
  - Middle: α=1e³ (moderately stiff) — differences appear
  - Right: α=1e⁶ (very stiff) — TR fails, TRBDF2 succeeds
- **Stiffness demo fix**: Split into 3 subplots showing stable vs. unstable time steps
  - Clearly shows red line (unstable) diverging for α=1e⁶
- **Result**: GPU is **0.04× slower** due to small problem size (overhead-bound, not compute-bound)

## Running Problems

### Prerequisites
```bash
module load cuda
pip install cupy numpy scipy matplotlib
```

### Individual Problems

```bash
# Problem 1: Convergence and timing for exponential decay
sbatch run_problem1.sh
python hw6_cupy.py problem1 --outdir results/problem1

# Problem 2: Complex dynamics analysis
sbatch run_problem2.sh
python hw6_cupy.py problem2 --outdir results/problem2 --grid-n 801

# Problem 3: Batched logistic trajectories (GPU-accelerated)
sbatch run_problem3.sh
python hw6_cupy.py problem3 --outdir results/problem3 --Ns 1000 10000 100000 1000000

# Problem 4: Stiff system analysis (implicit methods)
sbatch run_problem4.sh
python hw6_cupy.py problem4 --outdir results/problem4 --d 1000000

# Run all problems
sbatch hw6_cupy.py all --outdir results
```

## Key Insights

### When GPU Wins
| Problem | GPU Speedup | Why |
|---------|------------|-----|
| Problem 3 (N=10⁶) | **17×** | Embarrassingly parallel, compute-bound |
| Problem 3 (N=100K) | **3.4×** | Break-even point for overhead amortization |
| Problem 3 (N=1K) | **0.04×** | Overhead dominates, underutilized cores |
| Problem 4 | **0.04×** | Light computation, memory transfer dominates |

### Overhead Considerations
- **Kernel launch**: ~1-2 ms per GPU operation
- **Memory transfer**: Significant for small N or light compute
- **Occupancy**: GPU needs sufficient parallelism to hide latency
- **Saturation**: GPU fully utilized at large batch sizes

### Stiffness & Stability
- **Explicit Euler**: Unstable when |1 - dt·α| > 1
- **TR (A-stable)**: Stable but doesn't damp high-frequency modes
- **TRBDF2 (L-stable)**: Optimal for stiff systems, strong damping

## File Structure
```
hw6/src/
├── backend.py              # NumPy/CuPy backend abstraction
├── hw6_cupy.py            # Main solver implementations
├── run_problem[1-4].sh    # SLURM job scripts
└── results/
    ├── problem1/          # Convergence & timing results
    ├── problem2/          # Frequency domain plots
    ├── problem3/          # Speedup curves (GPU wins!)
    └── problem4/          # Stiffness & stability plots (subplots)
```

## SLURM Configuration
```bash
#SBATCH --partition=cmt          # Compute node partition
#SBATCH --gpus-per-node=1        # 1 GPU per node
#SBATCH --time=01:00:00          # 1 hour walltime
#SBATCH --mem=16GB               # 16 GB memory
```

## Monitoring Jobs
```bash
# Submit and get JOB_ID
JOB_ID=$(sbatch run_problem3.sh | awk '{print $4}')

# Check job status
scontrol show job $JOB_ID | grep -E "ElapsedTime|TimeLimit"

# Monitor in real-time
watch -n 2 "squeue -j $JOB_ID"

# View logs
tail -f problem3_${JOB_ID}.log
```

## Notes
- Problem 3 CPU benchmarking is the bottleneck for large N—consider skipping CPU for N≥10⁶
- Problem 4 visualizations now use subplots for clarity (not overlapping lines)
- GPU benefits require sufficient parallelism (N≥50K for problem 3)
- Stiff systems require implicit methods (TRBDF2 > TR > Explicit Euler)
