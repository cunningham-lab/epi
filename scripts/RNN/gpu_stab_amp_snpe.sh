#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=gpu_snpe
#SBATCH --gres=gpu:v100:1
#SBATCH -c 1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=50gb

module load cuda11.0/toolkit
module load cuda11.0/blas
module load cudnn8.0-cuda11.0

source activate sbi_gpu
python3 stab_amp_snpe.py --N $1 --num_sims $2 --num_batch $3 --num_transforms $4 --num_atoms $5 --g $6 --K $7 --rs $8 --max_rounds 100
