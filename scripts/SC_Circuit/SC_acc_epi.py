import os
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.SC_Circuit_4 import SC_acc
import time

DTYPE = tf.float32

# Parse script command-line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=float, default=0.5) # neuron type
parser.add_argument('--beta', type=float, default=4.) # aug lag hp
parser.add_argument('--logc0', type=float, default=0.) # log10 of c_0
parser.add_argument('--bnmom', type=float, default=.99) # log10 of c_0
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

p = args.p
AL_beta = args.beta
c0 = 10.**args.logc0
bnmom = args.bnmom
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

model = Model("SC_Circuit", parameters)

# EP values
#mu_std = 0.1
mu = np.array([p, 1.-p])

model.set_eps(SC_acc)

#init_type = 'abc'
#abc_std = 0.025
#init_params = {'num_keep':100, 
#               'means':np.array([p, 1-p]),
#               'stds':np.array([abc_std, abc_std]),
#              }

# 3. Run EPI.
q_theta, opt_data, epi_path, failed = model.epi(
    mu,
    arch_type='coupling',
    num_stages=3,
    num_layers=2,
    num_units=50,
    post_affine=True,
    batch_norm=True,
    bn_momentum=bnmom,
    K=10,
    N=M,
    num_iters=2500,
    lr=1e-3,
    c0=c0,
    beta=AL_beta,
    nu=0.5,
    random_seed=random_seed,
    #init_type=init_type,
    #init_params=init_params,
    verbose=True,
    stop_early=True,
    log_rate=50,
    save_movie_data=True,
)

if not failed:
    print("Making movie.")
    model.epi_opt_movie(epi_path)
    print("done.")
