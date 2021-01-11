#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for alpha in E
do
  for beta in 2.
  do
    for epsilon in 0.25
    do
      for bnmom in 0.5 0.999
      do
        for rs in 1
        do
          sbatch cpu_drdh.sh $alpha $beta $epsilon 0 $bnmom $rs
        done
      done
    done
  done
done
