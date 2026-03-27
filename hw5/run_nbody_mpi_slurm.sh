#!/bin/bash

#SBATCH --job-name=nbody_mpi
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=nbody_mpi_%j.log
#SBATCH --error=nbody_mpi_%j.err

module load gcc

OUTPUT_FILE="${OUTPUT_FILE:-nbody_mpi_results.txt}"
N_VALUES=(1024 2048 4096 8192)
RANK_VALUES=(1 2 4)
MAX_RANKS="${SLURM_NTASKS:-4}"

echo "Compiling nbody_mpi.cc..."
mpicxx -O3 -std=c++17 -o nbody_mpi nbody_mpi.cc -lm

> "${OUTPUT_FILE}"

echo "Running warmup..."
srun --ntasks=1 --cpus-per-task=1 ./nbody_mpi 128 > /dev/null
echo ""

for n in "${N_VALUES[@]}"; do
    for ranks in "${RANK_VALUES[@]}"; do
        if [ "${ranks}" -gt "${MAX_RANKS}" ]; then
            echo "Skipping N=${n} with ${ranks} ranks (exceeds allocated tasks)"
            continue
        fi

        echo "Running N=${n} with ${ranks} ranks..."
        START=$(date +%s.%N)
        srun --ntasks="${ranks}" --cpus-per-task=1 ./nbody_mpi "${n}" > /dev/null
        END=$(date +%s.%N)
        RUNTIME=$(echo "${END} - ${START}" | bc)
        echo "${n},${ranks},${RUNTIME}" >> "${OUTPUT_FILE}"
        echo "  Runtime: ${RUNTIME}s"
    done
done

echo ""
echo "Results saved to ${OUTPUT_FILE}"
