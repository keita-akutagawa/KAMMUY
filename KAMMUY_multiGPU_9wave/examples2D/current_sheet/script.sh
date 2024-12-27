#!/bin/bash
#SBATCH --partition=ga40-1gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j

module load nvhpc

export OMPI_MCA_orte_base_help_aggregate=0
export OMPI_MCA_opal_cuda_support=0

mpiexec -n 1 ./program

