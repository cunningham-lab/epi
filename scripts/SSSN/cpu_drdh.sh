#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=dvdh_epi
#SBATCH -c 1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=5gb

source activate epi
python3 dvdh_epi.py --alpha $1 --beta $2 --contrast $3 --logc0 $4 --random_seed $5
