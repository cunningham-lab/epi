#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=stab_amp_epi
#SBATCH -c 1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=10gb

source activate epi
python3 stab_amp_epi.py --N $1 --c0 $2 --rs $3
