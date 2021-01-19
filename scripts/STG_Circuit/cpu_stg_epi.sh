#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=STG_epi
#SBATCH -c 1
#SBATCH --time=11:00:00
#SBATCH --mem-per-cpu=50gb

source activate epi
python3 stg_epi_fast.py --freq $1 --mu_std $2 --elemwise_fn $3 --num_layers $4 --logc0 $5 --random_seed $6
