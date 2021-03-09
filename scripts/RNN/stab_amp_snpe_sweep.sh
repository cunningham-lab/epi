#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for N in 2 5
do
  for n_train in 100 1000
  do
    for n_mades in 1 2
    do
      for n_atoms in 25 100
      do
        for rs in {1..3}
        do
          echo "Running N = $N, n_train $n_train, n_mades $n_mades, n_atoms = $n_atoms, rs = $rs"
          sbatch gpu_stab_amp_snpe.sh $N $n_train $n_mades $n_atoms 0 1 $rs
        done
      done
    done
  done
done
