#!/bin/bash
source activate epi

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for n in 2 4
do
  for T in 10
  do
    for traj in 0
    do
      sbatch rnn_epi_timing.sh $n $T $traj
    done
  done
done
