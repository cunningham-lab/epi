#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=sa_smc
#SBATCH -c 1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=1gb

source activate epi
python3 stab_amp_smc.py --N $1 --rs $2
