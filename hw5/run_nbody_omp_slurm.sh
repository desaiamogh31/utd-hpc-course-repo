#!/bin/bash

#SBATCH --job-name=nbody_omp_benchmark
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=nbody_omp_benchmark_%j.log
#SBATCH --error=nbody_omp_benchmark_%j.err


# Load necessary modules
module load gcc

# Configuration
OUTPUT_FILE="nbody_omp_runtimes_ganymede2.txt"
N_VALUES=(1024 2048 4096 8192)
THREAD_VALUES=(1 2 4 8 16 32)

# Compile the program once
echo "Compiling nbody_omp..."
g++ -fopenmp -O3 -std=c++17 -o nbody_omp nbody_omp.cc -lm

# Clear output file
> "$OUTPUT_FILE"

# Warmup run to prime caches
echo "Running warmup..."
export OMP_NUM_THREADS=1
./nbody_omp 128 > /dev/null
echo ""

# Benchmark different N values and thread counts
for n in "${N_VALUES[@]}"; do
    for threads in "${THREAD_VALUES[@]}"; do
        # Check if thread count exceeds available CPUs
        if [ $threads -gt 32 ]; then
            echo "Skipping N=$n with $threads threads (exceeds available CPUs)"
            continue
        fi
        
        echo "Running N=$n with $threads threads..."
        export OMP_NUM_THREADS=$threads
        
        # Capture start time
        START=$(date +%s.%N)
        
        # Run the benchmark
        ./nbody_omp "$n" > /dev/null
        
        # Capture end time
        END=$(date +%s.%N)
        
        # Calculate runtime
        RUNTIME=$(echo "$END - $START" | bc)
        
        # Save results
        echo "$n,$threads,$RUNTIME" >> "$OUTPUT_FILE"
        echo "  Runtime: ${RUNTIME}s"
    done
done

echo ""
echo "Results saved to $OUTPUT_FILE"
