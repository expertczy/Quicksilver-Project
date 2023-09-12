#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=08:00:00

module purge
module load intel/2022a

for i in {24,48,72,96,192}
do
    srun -n$i ./qs
done