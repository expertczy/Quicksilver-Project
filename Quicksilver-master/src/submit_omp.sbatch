#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3700
#SBATCH --time=08:00:00

module purge
module load intel/2022a



export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for i in {100000,300000,500000,1000000,1500000}
do
    srun ./qs --nParticles=$i
done