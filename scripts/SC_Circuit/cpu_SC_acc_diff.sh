#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=SC_acc_epi
#SBATCH -c 1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=50gb

source activate epi
python3 SC_acc_diff.py --beta $1 --logc0 $2 --bnmom $3 --random_seed $4
