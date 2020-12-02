#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=STG_epi
#SBATCH -c 1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=50gb

source activate epi
python3 stg_epi_fast.py --freq $1 --beta $2 --logc0 $3 --random_seed $4
