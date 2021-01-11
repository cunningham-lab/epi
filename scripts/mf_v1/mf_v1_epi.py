"""Run EPI on oscillating 2D LDS. """

from epi.models import Model, Parameter
from epi.mf_v1 import mean_field_EI
import numpy as np
import argparse

# Get random seed.
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('--c0', type=float)
args = parser.parse_args()

print('Running epi on MF V1 with hyper parameter random seed %d.' % args.seed)

# Define the mean-field V1 model parameters.  
lb_W = np.array([0., -2., 0., -2.])
ub_W = np.array([2.,  0., 2.,  0.])
lb_sigma = np.array([0., 0., 0., 0.])
ub_sigma = 0.25*np.array([1., 1., 1., 1.])

W = Parameter("W", D=4, lb=lb_W, ub=ub_W)
sigma = Parameter("sigma", D=4, lb=lb_sigma, ub=ub_sigma)
parameters = [W, sigma]

M = Model("mf_v1", parameters)

# Set the emergent property statistics to frequency.
M.set_eps(mean_field_EI)

# Set the mergent property values
mu = np.array([0.6949108,
               0.92284465,
               1.0795053,
               0.80292356,
               1.1306603,
               1.402117,
               0.2737772,
               0.32565317,
               0.42739636,
               0.27382335,
               0.3258072,
               0.42774484])


np.random.seed(args.seed)
c0_exp = args.c0
num_stages = 3
num_layers = 2
num_units = 20

q_theta, opt_data, save_path, failed = M.epi(
    mu, 
    arch_type='coupling', 
    num_stages=num_stages,
    num_layers=num_layers,
    num_units=num_units,
    post_affine=False,
    K=12, 
    num_iters=2500, 
    N=500,
    lr=1e-3, 
    c0=10.**c0_exp, 
    verbose=True,
    log_rate=50,
    save_movie_data=True,
    random_seed=args.seed,
)
print("EPI done.")
print("Saved to %s." % save_path)
