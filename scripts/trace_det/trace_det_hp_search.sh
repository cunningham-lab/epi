#!/bin/bash
source activate epi

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for d in 10
do
  for hp_rs in {1..5}
  do
      echo "Running d = $d , hp seed = $hp_rs"
      sbatch gpu_trace_det_epi.sh $d $hp_rs
  done
done
