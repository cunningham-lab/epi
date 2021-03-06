#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=linear_2D
#SBATCH -c 1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2gb

source activate epi
python3 lds_epi.py --seed $1
