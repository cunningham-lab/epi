#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=drdh_epi
#SBATCH -c 1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=5gb

source activate epi
python3 drdh_epi.py --alpha $1 --beta $2 --epsilon $3 --logc0 $4 --bnmom $5 --random_seed $6
