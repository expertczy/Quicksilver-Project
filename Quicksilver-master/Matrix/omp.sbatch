#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3700
#SBATCH --time=00:10:00

module purge && module load intel/2022a

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# srun ./omp

# for i in {1..24}
# do
#     export OMP_NUM_THREADS=$i
#     srun ./omp -n 2000
# done

# for i in {500,1000,1500,2000,2500}
# do
#     export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
#     srun ./omp -n $i
# done