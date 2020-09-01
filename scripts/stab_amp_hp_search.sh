#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for N in 50 100
do
  echo "Running N = $N"
  sbatch cpu_stab_amp_epi.sh $N
done
