"""Run EPI on oscillating 2D LDS. """

from epi.models import Model, Parameter
from epi.example_eps import linear2D_freq
from epi.util import sample_aug_lag_hps
import numpy as np
import argparse

# Get random seed.
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int)
args = parser.parse_args()

print('Running epi on 2D-LDS with hyper parameter random seed %d.' % args.seed)

# Define the 2D LDS model parameters.  
# The four entries of the dynamics matrix will be bounded.
bounds = [-10., 10.]
a11 = Parameter("a11", bounds)
a12 = Parameter("a12", bounds)
a21 = Parameter("a21", bounds)
a22 = Parameter("a22", bounds)
params = [a11, a12, a21, a22]
M = Model("lds", params)

# Set the emergent property statistics to frequency.
M.set_eps(linear2D_freq)

# Set the mergent property values
mu = np.array([0.0, 0.25**2, 2 * np.pi, (0.2 * np.pi)**2])

np.random.seed(args.seed)
aug_lag_hps = sample_aug_lag_hps(1, c0_bounds=[1e-4, 1e-1])

init_params = {'loc':0., 'scale':3.}
q_theta, opt_data, save_path = M.epi(
    mu, 
    arch_type='autoregressive', 
    num_stages=1, 
    num_layers=2,
    num_units=15,
    post_affine=True,
    init_params=init_params,
    K = 2, 
    num_iters=1000, 
    N=aug_lag_hps.N, 
    lr=aug_lag_hps.lr, 
    c0=aug_lag_hps.c0, 
    gamma=aug_lag_hps.gamma,
    beta=aug_lag_hps.beta,
    verbose=True,
    log_rate=50,
    save_movie_data=True,
)
print("EPI done.")
print("Saved to %s." % save_path)
print("Writing movie...")
M.epi_opt_movie(save_path)
print("done.")
