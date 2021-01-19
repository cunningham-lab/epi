#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=rnn_timing
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=100gb

module load cuda90/toolkit
module load cuda90/blas
module load cudnn/7.0.5

source activate epi_gpu
python3 rnn_epi_timing.py --n $1 --T $2 --traj $3
