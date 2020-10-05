#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=SV_flip_epi
#SBATCH -c 1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=5gb

source activate epi
python3 SV_flip_epi.py --logc0 $1 --random_seed $2
