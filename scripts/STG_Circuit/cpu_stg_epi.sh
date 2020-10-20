#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=STG_epi
#SBATCH -c 1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=50gb

source activate epi
python3 stg_epi_fast.py --beta $1 --logc0 $2 --bnmom $3 --random_seed $4
