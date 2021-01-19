#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for beta in 2. 4.
do
  for logc0 in 3
  do
    for bnmom in 0.99
    do
      for rs in 1 2
      do
        sbatch cpu_SC_acc_diff.sh $beta $logc0 $bnmom $rs
      done
    done
  done
done
