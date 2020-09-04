#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=mat_det
#SBATCH -c 1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=2gb

source activate epi
python3 trace_det_epi.py --d $1 --seed $2
