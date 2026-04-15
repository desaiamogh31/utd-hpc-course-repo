#!/bin/bash
#SBATCH --job-name=hw6_problem2
#SBATCH --partition=cmt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --output=problem2_%j.log
#SBATCH --error=problem2_%j.err

module load cuda

# Run problem 2
python hw6_cupy.py problem2 --outdir results/problem2
