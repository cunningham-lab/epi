#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=SC_acc_epi
#SBATCH -c 1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=50gb

source activate epi
python3 SC_epi.py --p $1 --beta $2 --elemwise_fn $3 --logc0 $4 --mu_std $5 --random_seed $6
