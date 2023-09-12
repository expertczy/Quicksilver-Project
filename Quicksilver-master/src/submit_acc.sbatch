#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3700
#SBATCH --time=00:05:00

module purge && module load NVHPC/23.1-CUDA-12.0.0


# srun ./qs --nParticles=1000000

for i in {100000,300000,500000,1000000,1500000}
do
    export ACC_NUM_CORES=1
    ./qs --nParticles=$i
done

for i in {100000,300000,500000,1000000,1500000}
do
    export ACC_NUM_CORES=${SLURM_CPUS_PER_TASK}
    ./qs --nParticles=$i
done