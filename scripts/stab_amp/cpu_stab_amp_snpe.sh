#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=sa_snpe
#SBATCH -c 1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=2gb

source activate delfi
python3 stab_amp_snpe.py --N $1 --n_train 10000 --n_mades 2 --n_atoms $2 --rs $3
