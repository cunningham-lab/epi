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
    conv_inds = opt_data['converged'] == True
    if (sum(conv_inds) == 0):
        continue
    Hs = opt_data[conv_inds]['H']
    max_H_i = Hs.max()
    if max_H_i > max_H:
        max_H = max_H_i
        max_H_opt = opt_dir

print("Best opt was %s with entropy %.2E." % (max_H_opt, max_H))
    
  
