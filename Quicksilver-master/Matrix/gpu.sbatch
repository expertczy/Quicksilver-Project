#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1

module purge && module load NVHPC/23.1-CUDA-12.0.0
# srun ./gpu -n 2000

echo -e "size \n"
for i in {500,1000,1500,2000,2500}
do
    srun ./gpu -n $i
done