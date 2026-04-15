#!/bin/bash
#SBATCH --job-name=hw6_problem1
#SBATCH --partition=cmt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --output=problem1_%j.log
#SBATCH --error=problem1_%j.err

# Load required modules (adjust based on your HPC system)
# module load python/3.10
# module load cuda/11.8
# module load gcc/11.2

# Set up environment
module load cuda

# Run only problem 1
python hw6_cupy.py problem1 --outdir results/problem1