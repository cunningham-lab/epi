#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for freq in 0.55
  do
  for mu_std in 0.025 0.05
  do
    for g_el_lb in 0.01 4.
    do
      for beta in 4.
      do
        for logc0 in 2
        do
          for rs in 1
          do
            sbatch cpu_stg_epi.sh $freq $mu_std $g_el_lb $beta $logc0 $rs
          done
        done
      done
    done
  done
done
