import os
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.SSSN import SSSN_sim_sigma_c0, SSSN_sim_sigma_c1, load_SSSN_variable

# Parse script command-line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--contrast', type=float, default=0.) # aug lag hp
parser.add_argument('--alpha', type=str, default='E') # neuron type
parser.add_argument('--beta', type=float, default=4.) # aug lag hp
parser.add_argument('--logc0', type=float, default=0.) # log10 of c_0
parser.add_argument('--bnmom', type=float, default=.99) # log10 of c_0
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

contrast = args.contrast
alpha = args.alpha
beta = args.beta
c0 = 10.**args.logc0
bnmom = args.bnmom
random_seed = args.random_seed

neuron_inds = {'E':0, 'P':1, 'S':2, 'V':3}
neuron_ind = neuron_inds[alpha]

M = 200

# 1. Specify the V1 model for EPI.
D = 4
lb = np.zeros((D,))
ub = 1.*np.ones((D,))

sigma = Parameter("sigma", D, lb=lb, ub=ub)

# Define model
name = "SSSN_drdh_sigma_%s_c=%.1f" % (alpha, contrast)
parameters = [sigma]
model = Model(name, parameters)

if contrast == 0.:
    x1 = SSSN_sim_sigma_c0(np.zeros((1,4)))[:,:,neuron_ind]
    def dr(sigma):
        x2 = SSSN_sim_sigma_c0(sigma)[:,:,neuron_ind]
        diff = tf.reduce_mean(x2 - x1, axis=1)
        T_x = tf.stack((diff, diff ** 2), axis=1)
        return T_x

elif contrast == 1.:
    x1 = SSSN_sim_sigma_c1(np.zeros((1,4)))[:,:,neuron_ind]
    def dr(sigma):
        x2 = SSSN_sim_sigma_c1(sigma)[:,:,neuron_ind]
        diff = tf.reduce_mean(x2 - x1, axis=1)
        T_x = tf.stack((diff, diff ** 2), axis=1)
        return T_x
else:
    print('Error: Contrast must be 0 or 1')
    exit()

model.set_eps(dr)

# Emergent property values.
mu_std = 0.025
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
    bn_momentum=bnmom,
    K=5,
    N=M,
    num_iters=2000,
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
    print("Making movie.")
    model.epi_opt_movie(epi_path)
    print("done.")
