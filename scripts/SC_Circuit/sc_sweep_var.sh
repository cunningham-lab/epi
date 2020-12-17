#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for p in 0.75
do
  for beta in 4.
  do
    for logc0 in 2
    do
      for mu_std in 0.05
      do
        for rs in 1 2 3 4
        do
          sbatch cpu_SC_acc_epi_var.sh $p $beta $logc0 $mu_std $rs
        done
      done
    done
  done
done
