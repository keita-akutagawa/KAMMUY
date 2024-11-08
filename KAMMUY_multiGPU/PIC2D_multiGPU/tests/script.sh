#!/bin/bash
#SBATCH --partition=ga40-2gpu
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j

module load nvhpc

export OMPI_MCA_orte_base_help_aggregate=0
export OMPI_MCA_opal_cuda_support=0

mpiexec -n 2 ./program

