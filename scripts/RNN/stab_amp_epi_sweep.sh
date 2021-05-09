#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for N in 250
do
  for g in 0.01
  do
    for K in 1
    do
      for rs in 1 2
      do
        echo "Running N = $N, g = $g, K = $K, rs = $rs"
        sbatch cpu_stab_amp_epi.sh $N $g $K $rs
      done
    done
  done
done
