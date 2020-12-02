#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for freq in 0.53
  do
  for beta in 4.
  do
    for logc0 in 2
    do
      for rs in 1 2 3
      do
        sbatch cpu_stg_epi.sh $freq $beta $logc0 $rs
      done
    done
  done
done
