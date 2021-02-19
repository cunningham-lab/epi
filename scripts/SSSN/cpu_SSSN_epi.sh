#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=drdh_epi
#SBATCH -c 1
#SBATCH --time=11:30:00
#SBATCH --mem-per-cpu=25gb

source activate epi
python3 SSSN_epi.py --alpha E --ind $1 --sE_mean $2 --sE_std $3 --beta $4 --logc0 $5 --random_seed $6
