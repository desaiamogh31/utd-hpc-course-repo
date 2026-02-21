#!/bin/bash
#SBATCH --job-name=eigen-sweep
#SBATCH --output=eigen-%a_%A.out
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-17
#SBATCH --partition=cpu-preempt

# Optional: Load modules
# module load python
# conda activate hpc-s26

set -euo pipefail

# Sweep values
Ns=(10 20 40 80 160 320)
potentials=(well harmonic anisotropic_harmonic)

nN=${#Ns[@]}
np=${#potentials[@]}
ntasks=$((nN*np))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$ntasks" ]; then
        echo "Invalid TASK_ID: $SLURM_ARRAY_TASK_ID exceeds total tasks $ntasks"
        exit 1
fi

iN=$(( SLURM_ARRAY_TASK_ID / np ))
ip=$(( SLURM_ARRAY_TASK_ID % np ))

N=${Ns[$iN]}
pot=${potentials[$ip]}

mkdir -p results

python3 sparse_eigen.py --N "$N" --potential "$pot" --n-eigs 10 --out "results/eigs_N${N}_${pot}.txt"
