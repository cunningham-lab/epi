#!/bin/bash
source activate epi

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
<<<<<<< Updated upstream:scripts/rnn_epi_timing_batch.sh
for n in 2 4
=======
for N in 7
>>>>>>> Stashed changes:scripts/stab_amp/stab_amp_smc_sweep.sh
do
  for T in 10
  do
    for traj in 0
    do
      sbatch rnn_epi_timing.sh $n $T $traj
    done
  done
done
