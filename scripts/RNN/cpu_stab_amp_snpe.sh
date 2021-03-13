#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=rnn_snpe
#SBATCH -c 1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=1gb

echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
source activate sbi
python3 stab_amp_snpe.py --N $1 --num_sims $2 --num_batch $3 --num_transforms $4 --num_atoms $5 --g $6 --K $7 --rs $8
