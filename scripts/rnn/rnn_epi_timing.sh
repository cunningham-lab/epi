#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=rnn_timing
#SBATCH -c 1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=2gb

source activate epi
python3 rnn_epi_timing.py --n $1 --T $2 --traj $3
