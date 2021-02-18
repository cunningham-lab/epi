#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for ind in 49
do
  for sE_mean in 1.
  do
    for sE_std in 0.125
    do
      for beta in 2. 4.
      do
        for logc0 in 1 2
        do
          for rs in 1 2 3
          do
            sbatch cpu_SSSN_epi.sh $ind $sE_mean $sE_std $beta $logc0 $rs
          done
        done
      done
    done
  done
done
