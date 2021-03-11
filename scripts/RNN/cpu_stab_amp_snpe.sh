#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=rnn_snpe
#SBATCH -c 1
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1gb

source activate sbi
python3 stab_amp_snpe.py --N $1 --num_batch $2 --num_transforms $3 --num_atoms $4 --g $5 --K $6 --rs $7
