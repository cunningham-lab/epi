#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=gpu_epi
#SBATCH --gres=gpu:v100:1
#SBATCH -c 1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=50gb

module load cuda11.0/toolkit
module load cuda11.0/blas
module load cudnn8.0-cuda11.0

source activate epi_gpu
python3 stab_amp_epi.py --N $1 --g $2 --K $3 --rs $4
