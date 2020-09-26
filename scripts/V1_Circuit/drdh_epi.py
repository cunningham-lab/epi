import os
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.example_eps import V1_dr_eps


# Parse script command-line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=str, default='E') # log10 of c_0
parser.add_argument('--epsilon', type=float, default=0.) # log10 of c_0
parser.add_argument('--logc0', type=float, default=0.) # log10 of c_0
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

alpha = args.alpha
epsilon = args.epsilon
c0 = 10.**args.logc0
random_seed = args.random_seed

# 1. Specify the V1 model for EPI.
D = 4
lb = -5.*np.ones((D,))
ub = 5.*np.ones((D,))

dh = Parameter("dh", D, lb=lb, ub=ub)

# Define model
name = "V1Circuit_eps=%.2f" % epsilon
parameters = [dh]
model = Model(name, parameters)

X_INIT = tf.constant(np.random.normal(1.0, 0.01, (1, 4, 1)).astype(np.float32))

inc_val = 0.
dr = V1_dr_eps(alpha, inc_val, epsilon)
model.set_eps(dr)

# Emergent property values.
mu = np.array([inc_val, 0.25**2])

# 3. Run EPI.
q_theta, opt_data, epi_path, failed = model.epi(
    mu,
    arch_type='coupling',
    num_stages=3,
    num_layers=2,
    num_units=50,
    post_affine=True,
    batch_norm=True,
    K=4,
    N=500,
    num_iters=1000,
    lr=1e-3,
    c0=c0,
    beta=4.,
    nu=0.5,
    random_seed=random_seed,
    verbose=True,
    stop_early=True,
    log_rate=100,
    save_movie_data=True,
)

if not failed:
    print("Making movie.")
    model.epi_opt_movie(epi_path)
    print("done.")
