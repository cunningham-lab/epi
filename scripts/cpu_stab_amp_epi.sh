#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=cpu_stab_amp
#SBATCH -c 1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=10gb

source activate epi
python3 stab_amp_epi.py --N $1
