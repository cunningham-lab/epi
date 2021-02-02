#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for alpha in P
do
  for beta in 2.
  do
    for logc0 in 0
    do
      for bnmom in 0.999
      do
        for rs in {1..5}
        do
          sbatch cpu_drdh.sh $alpha $beta 0.4 $logc0 $bnmom $rs
        done
      done
    done
  done
done
