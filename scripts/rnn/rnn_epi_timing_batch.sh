#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for n in 50
do
  for T in 20 200
  do
    for traj in 1
    do
      sbatch rnn_epi_timing.sh $n $T $traj
    done
  done
done
