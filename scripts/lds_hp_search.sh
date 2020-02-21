#!/bin/bash
source activate epi

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for hp_rs in {1..4}
do
    echo "Running opt $hp_rs"
    sbatch lds_epi.sh $hp_rs
done


