#!/bin/bash

#SBATCH --job-name=nbody_mpi_shared
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=nbody_mpi_shared_%j.log
#SBATCH --error=nbody_mpi_shared_%j.err

MPI_HOME=/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.6
export PATH="$MPI_HOME/bin:$PATH"

OUTPUT_FILE="nbody_mpi_shared_runtimes.txt"
N_VALUES=(4096)
RANK_VALUES=(1 2 4)
MAX_RANKS="${SLURM_NTASKS:-4}"

echo "Compiling nbody_mpi_shared.cc..."
"$MPI_HOME/bin/mpicxx" -O3 -std=c++17 -o nbody_mpi_shared nbody_mpi_shared.cc -lm

> "$OUTPUT_FILE"

echo "Running warmup..."
"$MPI_HOME/bin/mpiexec" -n 1 ./nbody_mpi_shared 128 > /dev/null
echo ""

for n in "${N_VALUES[@]}"; do
    for ranks in "${RANK_VALUES[@]}"; do
        if [ "${ranks}" -gt "${MAX_RANKS}" ]; then
            echo "Skipping N=${n} with ${ranks} ranks (exceeds allocated tasks)"
            continue
        fi

        echo "Running N=${n} with ${ranks} ranks..."
        START=$(date +%s.%N)
        "$MPI_HOME/bin/mpiexec" -n "${ranks}" ./nbody_mpi_shared "${n}" > /dev/null
        END=$(date +%s.%N)
        RUNTIME=$(echo "${END} - ${START}" | bc)

        echo "${n},${ranks},${RUNTIME}" >> "$OUTPUT_FILE"
        echo "  Runtime: ${RUNTIME}s"
    done
done

echo ""
echo "Results saved to $OUTPUT_FILE"
