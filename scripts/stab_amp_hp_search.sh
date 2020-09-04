#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for N in 100 200 500
do
  for c0 in 3
  do
    echo "Running N = $N, c0 = $c0"
    sbatch cpu_stab_amp_epi.sh $N $c0
  done
done
