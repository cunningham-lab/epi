#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for N in 2 5 10 15 20 25
do
  for c0 in 3
  do
    for rs in {1..5}
    do
      echo "Running N = $N, c0 = $c0, rs = $rs"
      sbatch cpu_stab_amp_epi.sh $N $c0 $rs
    done
  done
done
