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
parser.add_argument('--freq', type=float, default=0.53) # frequency for mu
parser.add_argument('--beta', type=float, default=4.) # aug lag hp
parser.add_argument('--logc0', type=float, default=0.) # log10 of c_0
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

freq = args.freq
beta = args.beta
c0 = 10.**args.logc0
random_seed = args.random_seed

#sleep_dur = np.abs(args.logc0) + random_seed/5. + beta/3.
#print('short stagger sleep of', sleep_dur, flush=True)
#time.sleep(sleep_dur)

# 1. Specify the V1 model for EPI.
D = 2 
g_el = Parameter("g_el", 1, lb=4., ub=8.)
g_synA = Parameter("g_synA", 1, lb=0.1, ub=4.)

# Define model
name = "STG"
parameters = [g_el, g_synA]
model = Model(name, parameters)

# Emergent property values.
mu_std = 0.025
mu = np.array([freq, mu_std**2])

sigma_I = 2.5e-11
dt = 0.025
T = 200
network_freq = NetworkFreq(dt, T, sigma_I, mu)

model.set_eps(network_freq)

# 3. Run EPI.
q_theta, opt_data, epi_path, failed = model.epi(
    mu,
    arch_type='coupling',
    num_stages=3,
    num_layers=2,
    num_units=25,
    post_affine=True,
    batch_norm=True,
    bn_momentum=0.,
    K=3,
    N=200,
    num_iters=2500,
    lr=1e-3,
    c0=c0,
    beta=beta,
    nu=0.5,
    random_seed=random_seed,
    verbose=True,
    stop_early=True,
    log_rate=50,
    save_movie_data=True,
)

if not failed:
    print("Making movie.", flush=True)
    model.epi_opt_movie(epi_path)
    print("done.", flush=True)
