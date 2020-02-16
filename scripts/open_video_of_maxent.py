import numpy as np
import pandas as pd
import os
import sys

results_dir = sys.argv[1]

if not os.path.exists(results_dir):
    raise IOError("EPI optimization results directory %s does not exist." % results_dir)

opt_dirs = os.listdir(results_dir)
max_H = np.NINF
max_H_opt = None
for i, opt_dir in enumerate(opt_dirs):
    opt_data = pd.read_csv(results_dir + '/' + opt_dir + '/opt_data.csv')
    max_H_i = np.max(opt_data['H'])
    if max_H_i > max_H:
        max_H = max_H_i
        max_H_opt = opt_dir

print("Best opt was %s with entropy %.2E." % (max_H_opt, max_H))
print("How do we automatically open the saved video in this folder?")
print("Or should we just copy the video of the best hyper parameters to a separate S3 bucket or folder where John will await the results during the demo.")
    
  
