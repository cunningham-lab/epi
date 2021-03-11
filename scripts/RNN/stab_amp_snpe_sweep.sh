#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for N in 2 5
do
  for n_atoms in 10
  do
    for g in 0.0001
    do
      for K in 1
      do
        for rs in 1
        do
          echo "Running N = $N, n_atoms = $n_atoms, g = $g, K = $K, rs = $rs"
          sbatch cpu_stab_amp_snpe.sh $N 1000 3 $n_atoms $g $K $rs
        done
      done
    done
  done
done
