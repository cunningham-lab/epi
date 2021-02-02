#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for ind in 62
do
  for ff_mean in 0.5
  do
    for ff_std in 0.125
    do
      for beta in 4.
      do
        for logc0 in 0 3 6
        do
          for rs in 1 2 3
          do
            sbatch cpu_ff.sh $ind $ff_mean $ff_std $beta $logc0 $rs
          done
        done
      done
    done
  done
done
