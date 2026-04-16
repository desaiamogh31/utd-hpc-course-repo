#!/bin/bash
#SBATCH --job-name=hw6_problem4
#SBATCH --partition=cmt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --output=problem4_%j.log
#SBATCH --error=problem4_%j.err

module load cuda

# Run problem 4
# Optional: customize --d, --dt-small, --dt-large, and --tf as needed
# Default: d=10^6, dt_small=1e-6, dt_large=1e-2, tf=1.0
# Example: python hw6_cupy.py problem4 --outdir results/problem4 --d 1000000 --dt-small 1e-6 --dt-large 1e-2 --tf 1.0
python hw6_cupy.py problem4 --outdir results/problem4
