"""Run EPI on oscillating 2D LDS. """

import numpy as np
import argparse
from epi.models import Model, Parameter
from epi.util import sample_aug_lag_hps
from epi.example_eps import linear2D_freq

# Get random seed.
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

print('Running epi on 2D-LDS with hyper parameter random seed %d.' % args.seed)

# Define the 2D LDS model parameters.  
# The four entries of the dynamics matrix will be bounded.
lb = -10.
ub = 10.
a11 = Parameter("a11", 1, lb=lb, ub=ub)
a12 = Parameter("a12", 1, lb=lb, ub=ub)
a21 = Parameter("a21", 1, lb=lb, ub=ub)
a22 = Parameter("a22", 1, lb=lb, ub=ub)
params = [a11, a12, a21, a22]
M = Model("lds_2D", params)

# Set the emergent property statistics to frequency.
M.set_eps(linear2D_freq)

# Set the mergent property values
mu = np.array([0.0, 0.5**2, 2 * np.pi, (0.1 * 2 * np.pi)**2])

np.random.seed(args.seed)
num_stages = 3
num_layers = 2
num_units = 50

q_theta, opt_data, save_path, failed = M.epi(
    mu, 
    arch_type='coupling', 
    num_stages=num_stages,
    num_layers=num_layers,
    num_units=num_units,
    post_affine=True,
    K = 10, 
    num_iters=2500, 
    N=500,
    lr=1e-3, 
    c0=1e-3, 
    verbose=True,
    stop_early=True,
    log_rate=100,
    save_movie_data=True,
)
print("EPI done.")
print("Saved to %s." % save_path)
if not failed:
    print("Writing movie...")
    M.epi_opt_movie(save_path)
    print("done.")
