#!/bin/bash
#SBATCH -J hw4-lorentz
#SBATCH -p cpu-preempt
#SBATCH --nodes=1 # nodes
#SBATCH -n 16 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:10:00 # walltime
#SBATCH --exclusive # avoid interference
#SBATCH --output=parallel-output.txt
#SBATCH --mail-user=and230001@utdallas.edu

module purge
module load gnu12/12.4.0
module load mpich/3.4.3-ucx

# Avoid oversubscription by BLAS/NumPy threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mpirun -n 16 python mpi_lorentz.py --n 1000000 --bins 100
