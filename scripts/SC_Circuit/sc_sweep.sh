#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for p in 0.75
do
  for logc0 in -3 3
  do
    for bnmom in 0.99
    do
      for rs in 1 2
      do
        sbatch cpu_SC_acc_epi.sh $p 4. $logc0 $bnmom $rs
      done
    done
  done
done
