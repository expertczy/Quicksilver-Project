#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=3700
#SBATCH --time=08:00:00

module purge
module load intel/2022a

srun ./qs