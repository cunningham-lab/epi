#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for N in 20 25
do
  for n_train in 10000
  do
    for n_mades in 2
    do
      for n_atoms in 100
      do
        for rs in {1..5}
        do
          echo "Running N = $N, n_train $n_train, n_mades $n_mades, n_atoms = $n_atoms, rs = $rs"
          sbatch cpu_stab_amp_snpe.sh $N $n_train $n_mades $n_atoms $rs
        done
      done
    done
  done
done
