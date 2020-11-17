import os
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.SSSN import SSSN_sim, SSSN_sim, load_SSSN_variable

# Parse script command-line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--contrast', type=float, default=0.) # aug lag hp
parser.add_argument('--alpha', type=str, default='E') # neuron type
parser.add_argument('--beta', type=float, default=4.) # aug lag hp
parser.add_argument('--logc0', type=float, default=0.) # log10 of c_0
parser.add_argument('--eps', type=float, default=0.01) # log10 of c_0
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

contrast = args.contrast
alpha = args.alpha
beta = args.beta
c0 = 10.**args.logc0
eps = args.eps
random_seed = args.random_seed

ind = 1070
hb = load_SSSN_variable('hb', ind=ind)
hc = load_SSSN_variable('hc', ind=ind)
H = (hb + contrast*hc)[None,:]

neuron_inds = {'E':0, 'P':1, 'S':2, 'V':3}
neuron_ind = neuron_inds[alpha]

M = 500
eps = 0.1

# 1. Specify the V1 model for EPI.
D = 4
lb = -.25*np.ones((D,))
ub = .25*np.ones((D,))

dh = Parameter("dh", D, lb=lb, ub=ub)

# Define model
name = "SSSN_drdh_%s_c=%.1f_eps=%.3f" % (alpha, contrast, eps)
parameters = [dh]
model = Model(name, parameters)

sssn_sim = SSSN_sim(eps)

def dr(dh):
    x1 = sssn_sim(H+tf.zeros_like(dh))[:,:,neuron_ind]
    x2 = sssn_sim(H + dh)[:,:,neuron_ind]

    diff = tf.reduce_mean(x2 - x1, axis=1)
    T_x = tf.stack((diff, diff ** 2), axis=1)

    return T_x

model.set_eps(dr)

# Emergent property values.
mu_std = 0.05
mu = np.array([0., mu_std**2])

# 3. Run EPI.
q_theta, opt_data, epi_path, failed = model.epi(
    mu,
    arch_type='coupling',
    num_stages=3,
    num_layers=2,
    num_units=50,
    post_affine=True,
    batch_norm=True,
    bn_momentum=0.0,
    K=15,
    N=M,
    num_iters=2500,
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
