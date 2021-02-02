#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=drdh_epi
#SBATCH -c 1
#SBATCH --time=11:30:00
#SBATCH --mem-per-cpu=25gb

source activate epi
python3 ff_epi_sigma.py --alpha E --ind $1 --ff_mean $2 --ff_std $3 --beta $4 --logc0 $5 --random_seed $6
