#!/bin/bash
source activate epi

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for c0 in -10 -5 0 5 10
  do
  for hp_rs in {1..4}
  do
    echo "Running opt $c0 $hp_rs"
    sbatch mf_v1_epi.sh $c0 $hp_rs 
  done
done


