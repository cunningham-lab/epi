#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=drdh_epi
#SBATCH -c 1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=25gb

source activate epi
python3 drdh_epi.py --alpha $1 --beta $2 --contrast $3 --logc0 $4 --random_seed $5
