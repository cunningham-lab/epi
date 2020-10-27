#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=SC_acc_epi
#SBATCH -c 1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=50gb

source activate epi
python3 SC4_acc_epi.py --p $1 --beta $2 --logc0 $3 --bnmom $4 --random_seed $5
