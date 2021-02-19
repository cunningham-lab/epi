#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for freq in 0.55
do
  for mu_std in 0.025
  do
    for beta in 2.
    do
      for logc0 in 5
      do
        for rs in {6..15}
        do
          sbatch cpu_stg_epi.sh $freq $mu_std $beta $logc0 $rs
        done
      done
    done
  done
done
