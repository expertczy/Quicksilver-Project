#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=08:00:00

module purge
module load intel/2022a

MY_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=$MY_NUM_THREADS

srun ./qs