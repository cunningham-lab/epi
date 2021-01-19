#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=stab_amp_epi
#SBATCH -c 1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=2gb

source activate epi
python3 stab_amp_epi.py --N $1 --c0 $2 --rs $3
