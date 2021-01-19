#!/bin/bash

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for c0 in -2 0 2
do
  for rs in {1..3}
  do
    echo "Running SV Flip, c0 = $c0, rs = $rs"
    sbatch cpu_SV_flip.sh $c0 $rs
  done
done
