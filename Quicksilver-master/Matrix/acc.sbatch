#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3700
#SBATCH --time=08:00:00

module purge && module load NVHPC/23.1-CUDA-12.0.0

# export ACC_NUM_CORES=${SLURM_CPUS_PER_TASK}

# srun ./acc -n 2000
# echo -e "acc \n"
# echo -e "scale \n"
# for i in {1..24}
# do
#     export ACC_NUM_CORES=$i
#     srun ./acc -n 2000
# done

# echo -e "size \n"
for i in {500,1000,1500,2000,2500}
do
    # export ACC_NUM_CORES=8
    srun ./acc -n $i
done

# echo -e "serial \n"
# for i in {500,1000,1500,2000,2500}
# do
#     export ACC_NUM_CORES=1
#     srun ./acc -n $i
# done