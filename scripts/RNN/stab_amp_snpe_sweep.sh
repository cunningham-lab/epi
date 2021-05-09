#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for N in 250
do
  for g in 0.01
  do
    for K in 1
    do
      for num_sims in 250000
      do
        for num_batch in 1000
        do
          for num_atoms in 100
          do
            for rs in {1..5}
            do
              echo "Running N = $N, g = $g, K = $K, sims = $num_sims batch = $num_batch, atoms = $num_atoms, rs = $rs"
              sbatch cpu_stab_amp_snpe.sh $N $num_sims $num_batch 3 $num_atoms $g $K $rs
            done
          done
        done
      done
    done
  done
done
