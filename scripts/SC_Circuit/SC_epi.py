import os
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from neural_circuits.SC_Circuit_4 import SC_acc_var
import time

DTYPE = tf.float32

# Parse script command-line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=float, default=0.5) # neuron type
parser.add_argument('--beta', type=float, default=4.) # aug lag hp
parser.add_argument('--elemwise_fn', type=str, default="affine") # aug lag hp
parser.add_argument('--logc0', type=float, default=0.) # log10 of c_0
parser.add_argument('--mu_std', type=float, default=0.1) 
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

p = args.p
AL_beta = args.beta
elemwise_fn = args.elemwise_fn
c0 = 10.**args.logc0
mu_std = args.mu_std
random_seed = args.random_seed

M = 100
N = 200

# 1. Specify the V1 model for EPI.
lb = -5.
ub = 5.

sW = Parameter("sW", 1, lb=lb, ub=ub)
vW = Parameter("vW", 1, lb=lb, ub=ub)
dW = Parameter("dW", 1, lb=lb, ub=ub)
hW = Parameter("hW", 1, lb=lb, ub=ub)

parameters = [sW, vW, dW, hW]

model = Model("SC_Circuit_var", parameters)

# EP values
mu = np.array([p, 1.-p, mu_std**2, mu_std**2])

model.set_eps(SC_acc_var(p))

# 3. Run EPI.
q_theta, opt_data, epi_path, failed = model.epi(
    mu,
    arch_type='coupling',
    num_stages=3,
    num_layers=2,
    num_units=50,
    elemwise_fn=elemwise_fn,
    post_affine=False,
    batch_norm=False,
    bn_momentum=0.0,
    K=15,
    N=M,
    num_iters=2000,
    lr=1e-3,
    c0=c0,
    beta=AL_beta,
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
