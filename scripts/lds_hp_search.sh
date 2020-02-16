#!/bin/bash
source activate epi

# This shell script does it linearly, but could you run each of these 
# python scripts independently on different instances?
for hp_rs in {1..3}
do
    echo "Running opt $hp_rs"
    python3 lds_epi.py --seed $hp_rs
done

# Once all these instances are done running, collect the results.
echo "collecting results."
export resultsdir=data/lds_linear2D_freq_mu=0.00E+00_6.25E-02_6.28E+00_3.95E-01
python3 open_video_of_maxent.py $resultsdir


