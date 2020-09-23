#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=sa_snpe
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1gb

source activate delfi
module load gcc/7.2.0
module load cuda92/toolkit

THEANO_FLAGS=device=cuda python3 stab_amp_snpe.py --N $1 --n_train $2 --n_mades $3 --n_atoms $4 --rs $5
