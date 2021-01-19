import os
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.SSSN import SSSN_sim, SSSN_sim, load_SSSN_variable, get_stddev_sigma

# Parse script command-line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=str, default='E') # neuron type
parser.add_argument('--ind', type=int, default=62) # neuron type
parser.add_argument('--lim', type=float, default=0.005) # neuron type
parser.add_argument('--f_mean', type=float, default=5.) # neuron type
parser.add_argument('--f_std', type=float, default=0.25) # neuron type
parser.add_argument('--beta', type=float, default=4.) # aug lag hp
parser.add_argument('--logc0', type=float, default=0.) # log10 of c_0
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

alpha = args.alpha
ind = args.ind
lim = args.lim
f_mean = args.f_mean
f_std = args.f_std
beta = args.beta
c0 = 10.**args.logc0
random_seed = args.random_seed

contrast = 0.5
W_mat = load_SSSN_variable('W', ind=ind)
hb = load_SSSN_variable('hb', ind=ind).numpy()
hc = load_SSSN_variable('hc', ind=ind).numpy()
h = (hb[None,:] + contrast*hc[None,:])

neuron_inds = {'E':0, 'P':1, 'S':2, 'V':3}
neuron_ind = neuron_inds[alpha]

M = 100
# 1. Specify the V1 model for EPI.
D = 4
lb = np.zeros((D,))
ub = lim*np.ones((D,))
sigma_eps = Parameter("sigma_eps", D, lb=lb, ub=ub)

# Define model
name = "SSSN_stddev_sigma_%s_%.2E_%.2E_ind=%d" % (alpha, f_mean, f_std, ind)
parameters = [sigma_eps]
model = Model(name, parameters)

dt = 0.0005
T = 150
N = 100

stddev = get_stddev_sigma(alpha, W_mat, h, N=N, dt=dt, T=T, T_ss=T-50, mu=f_mean)
model.set_eps(stddev)

# Emergent property values.
mu = np.array([f_mean, f_std**2])

# 3. Run EPI.
q_theta, opt_data, epi_path, failed = model.epi(
    mu,
    arch_type='coupling',
    num_stages=3,
    num_layers=2,
    num_units=25,
    post_affine=True,
    batch_norm=False,
    bn_momentum=0.0,
    K=10,
    N=M,
    num_iters=2000,
    lr=1e-3,
    c0=c0,
    beta=beta,
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
