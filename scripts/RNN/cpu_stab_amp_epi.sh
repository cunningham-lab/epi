#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=cpu_epi
#SBATCH -c 1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=100gb

source activate epi
python3 stab_amp_epi.py --N $1 --g $2 --K $3 --rs $4 --c0 3
