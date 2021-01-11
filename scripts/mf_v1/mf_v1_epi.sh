#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=mf_v1
#SBATCH -c 1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=2gb

source activate epi
python3 mf_v1_epi.py --c0 $1 --seed $2
