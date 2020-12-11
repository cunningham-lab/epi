import os
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.STG_Circuit import NetworkFreq
import time

DTYPE = tf.float32

# Parse script command-line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--freq', type=float, default=0.55) # frequency for mu
parser.add_argument('--mu_std', type=float, default=0.05) # std in mu constraint
parser.add_argument('--g_el_lb', type=float, default=0.01) # lower bound on g_el
parser.add_argument('--beta', type=float, default=4.) # aug lag hp
parser.add_argument('--logc0', type=float, default=0.) # log10 of c_0
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

freq = args.freq
mu_std = args.mu_std
g_el_lb = args.g_el_lb
beta = args.beta
c0 = 10.**args.logc0
random_seed = args.random_seed

#sleep_dur = np.abs(args.logc0) + random_seed/5. + beta/3.
#print('short stagger sleep of', sleep_dur, flush=True)
#time.sleep(sleep_dur)

# 1. Specify the V1 model for EPI.
D = 2 
g_el = Parameter("g_el", 1, lb=g_el_lb, ub=8.)
g_synA = Parameter("g_synA", 1, lb=0.01, ub=4.)

# Define model
name = "STG"
parameters = [g_el, g_synA]
model = Model(name, parameters)

# Emergent property values.
mu = np.array([freq, mu_std**2])

init_type = 'abc'
abc_std = mu_std
init_params = {'num_keep':500,
               'means':np.array([freq]),
               'stds':np.array([abc_std,])}

sigma_I = 5e-13
dt = 0.025
T = 300
network_freq = NetworkFreq(dt, T, sigma_I, mu)

model.set_eps(network_freq)

# 3. Run EPI.
q_theta, opt_data, epi_path, failed = model.epi(
    mu,
    arch_type='coupling',
    num_stages=3,
    num_layers=2,
    num_units=25,
    #num_stages=0, # changed
    #num_layers=1, # changed
    #num_units=1, # changed
    post_affine=True,
    batch_norm=True,
    #batch_norm=False, # changed
    bn_momentum=0.,
    K=6,
    N=400,
    num_iters=5000,
    lr=1e-3,
    c0=c0,
    beta=beta,
    nu=0.5,
    random_seed=random_seed,
    init_type=init_type,
    init_params=init_params,
    verbose=True,
    stop_early=True,
    log_rate=50,
    save_movie_data=True,
)

if not failed:
    print("Making movie.", flush=True)
    model.epi_opt_movie(epi_path)
    print("done.", flush=True)
