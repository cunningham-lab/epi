#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=gpu_stab_amp
#SBATCH --gres=gpu
#SBATCH -c 1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=10gb

module load cuda90/toolkit
module load cuda90/blas
module load cudnn/7.0.5

source activate epi_gpu
python3 stab_amp_epi.py --N $1
