#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=rnn_snpe
#SBATCH -c 1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=2gb

source activate sbi
python3 stab_amp_snpe.py --N $1 --num_sims $2 --num_batch $3 --num_transforms $4 --num_atoms $5 --g $6 --K $7 --rs $8 --max_rounds 100
