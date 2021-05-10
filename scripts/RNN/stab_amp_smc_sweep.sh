#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for N in 5
do
  for rs in 3
  do
    echo "Running SMC, N = $N, rs = $rs"
    sbatch cpu_stab_amp_smc.sh $N $rs
  done
done
