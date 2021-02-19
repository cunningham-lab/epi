#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=STG_epi
#SBATCH -c 1
#SBATCH --time=11:00:00
#SBATCH --mem-per-cpu=50gb

source activate epi
python3 stg_epi.py --freq $1 --mu_std $2 --beta $3 --logc0 $4 --random_seed $5
