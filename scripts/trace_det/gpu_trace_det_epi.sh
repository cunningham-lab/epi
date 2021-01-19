#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=gpu_mat_det
#SBATCH --gres=gpu
#SBATCH -c 1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=2gb

module load cuda90/toolkit
module load cuda90/blas
module load cudnn/7.0.5

source activate epi_gpu
python3 trace_det_epi.py --d $1 --seed $2
