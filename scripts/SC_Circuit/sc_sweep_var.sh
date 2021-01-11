#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for p in 0.75
do
  for beta in 4.
  do
    for elemwise_fn in affine spline
    do
      for logc0 in 2
      do
        for mu_std in 0.05
        do
          for rs in 1
          do
            sbatch cpu_SC_acc_epi_var.sh $p $beta $elemwise_fn $logc0 $mu_std $rs
          done
        done
      done
    done
  done
done
