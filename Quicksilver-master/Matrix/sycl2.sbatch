#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=00:10:00

module purge && module load intel/2022a
# srun ./sycl2 -n 1000
echo -e "sycl size \n"
for i in {500,1000,1500,2000,2500}
do
    srun ./sycl2 -n $i
done