#!/bin/bash

#SBATCH --job-name=nbody_mpi_omp
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=nbody_mpi_omp_%j.log
#SBATCH --error=nbody_mpi_omp_%j.err

module load gcc

# Configuration
OUTPUT_FILE="nbody_mpi_omp_runtimes.txt"
N="${N:-4096}"
RANK_VALUES=(1 2 4)
THREAD_VALUES=(1 2 4 8)
MAX_RANKS="${SLURM_NTASKS:-4}"
MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "Compiling nbody_mpi_omp.cc..."
mpicxx -fopenmp -O3 -std=c++17 -o nbody_mpi_omp nbody_mpi_omp.cc -lm

# Clear output file
> "$OUTPUT_FILE"

# Warmup run to prime caches
echo "Running warmup..."
export OMP_NUM_THREADS=1
srun --ntasks=1 --cpus-per-task=1 ./nbody_mpi_omp 128 > /dev/null
echo ""

# Benchmark different rank counts and thread counts
for ranks in "${RANK_VALUES[@]}"; do
    if [ "${ranks}" -gt "${MAX_RANKS}" ]; then
        echo "Skipping N=${N} with ${ranks} ranks (exceeds allocated tasks)"
        continue
    fi

    for threads in "${THREAD_VALUES[@]}"; do
        if [ "${threads}" -gt "${MAX_THREADS}" ]; then
            echo "Skipping N=${N} with ${ranks} ranks and ${threads} threads (exceeds cpus per task)"
            continue
        fi

        echo "Running N=${N} with ${ranks} ranks and ${threads} threads..."
        export OMP_NUM_THREADS="${threads}"

        # Capture start time
        START=$(date +%s.%N)

        # Run the benchmark
        srun --ntasks="${ranks}" --cpus-per-task="${threads}" ./nbody_mpi_omp "${N}" > /dev/null

        # Capture end time
        END=$(date +%s.%N)

        # Calculate runtime
        RUNTIME=$(echo "${END} - ${START}" | bc)

        # Save results
        echo "${N},${ranks},${RUNTIME}" >> "$OUTPUT_FILE"
        echo "  Runtime: ${RUNTIME}s"
    done
done

echo ""
echo "Results saved to $OUTPUT_FILE"
