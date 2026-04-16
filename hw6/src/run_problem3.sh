#!/bin/bash
#SBATCH --job-name=hw6_problem3
#SBATCH --partition=cmt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --output=problem3_%j.log
#SBATCH --error=problem3_%j.err

module load cuda

# Run problem 3
# Optional: customize --Ns, --dt, and --tf as needed
python hw6_cupy.py problem3 --outdir results/problem3 --Ns 1000 10000 100000 1000000 
#python hw6_cupy.py problem3 --outdir results/problem3
