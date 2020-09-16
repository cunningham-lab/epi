#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=sa_snpe
#SBATCH -c 1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=1gb

source activate delfi
python3 stab_amp_snpe.py --N $1 --n_train $2 --n_mades $3 --n_atoms $4 --rs $5
