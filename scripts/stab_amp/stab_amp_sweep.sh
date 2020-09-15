#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for N in 2 4 6 8 10 15
do
  for n_atoms in 25
  do
    for rs in 1
    do
      echo "Running N = $N, n_atoms = $n_atoms, rs = $rs"
      sbatch cpu_stab_amp_snpe.sh $N $n_atoms $rs
    done
  done
done
