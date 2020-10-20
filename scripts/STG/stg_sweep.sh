#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for beta in 4.
do
  for logc0 in -3 0 3
  do
    for rs in 1 
    do
      sbatch cpu_stg_epi.sh $beta $logc0 $rs
    done
  done
done
