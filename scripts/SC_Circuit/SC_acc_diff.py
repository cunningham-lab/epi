import os
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
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

M = 200
N = 100

# 1. Specify the V1 model for EPI.
lb = -20.
ub = 20.

sW_P = Parameter("sW_P", 1, lb=lb, ub=ub)
sW_A = Parameter("sW_A", 1, lb=lb, ub=ub)

vW_PA = Parameter("vW_PA", 1, lb=lb, ub=ub)
vW_AP = Parameter("vW_AP", 1, lb=lb, ub=ub)

dW_PA = Parameter("dW_PA", 1, lb=lb, ub=ub)
dW_AP = Parameter("dW_AP", 1, lb=lb, ub=ub)

hW_P = Parameter("hW_P", 1, lb=lb, ub=ub)
hW_A = Parameter("hW_A", 1, lb=lb, ub=ub)

parameters = [sW_P, sW_A, vW_PA, vW_AP, dW_PA, dW_AP, hW_P, hW_A]

model = Model("SC_acc_diff", parameters)

# EP values
mu = np.array([0., 0.2])

t_cue_delay = 1.2
t_choice = 0.6
t_total = t_cue_delay + t_choice
dt = 0.024
t = np.arange(0.0, t_total, dt)
T = t.shape[0]

# input parameters
E_constant = 0.
E_Pbias = 0.
E_Prule = 1.
E_Arule = 1.
E_choice = 1.
E_light = 1.

opto_strength = 0.3
        
# set constant parameters
C = 4

theta = 0.05
beta = 0.5
tau = 0.09
sigma = 0.5

# inputs
I_constant = E_constant * tf.ones((T, 1, 1, 4, 1), dtype=DTYPE)

I_Pbias = np.zeros((T, 4))
I_Pbias[t < T * dt] = np.array([1, 0, 0, 1])
I_Pbias = I_Pbias[:,None,None,:,None]
I_Pbias = E_Pbias * tf.constant(I_Pbias, dtype=DTYPE)

I_Prule = np.zeros((T, 4))
I_Prule[t < 1.2] = np.array([1, 0, 0, 1])
I_Prule = I_Prule[:,None,None,:,None]
I_Prule = E_Prule * tf.constant(I_Prule, dtype=DTYPE)

I_Arule = np.zeros((T, 4))
I_Arule[t < 1.2] = np.array([0, 1, 1, 0])
I_Arule = I_Arule[:,None,None,:,None]
I_Arule = E_Arule * tf.constant(I_Arule, dtype=DTYPE)

I_choice = np.zeros((T, 4))
I_choice[t > 1.2] = np.array([1, 1, 1, 1])
I_choice = I_choice[:,None,None,:,None]
I_choice = E_choice * tf.constant(I_choice, dtype=DTYPE)

I_lightL = np.zeros((T, 4))
I_lightL[1.2 < t] = np.array([1, 1, 0, 0])
#I_lightL[np.logical_and(1.2 < t, t < 1.5)] = np.array([1, 1, 0, 0])
I_lightL = I_lightL[:,None,None,:,None]
I_lightL = E_light * tf.constant(I_lightL, dtype=DTYPE)

I_lightR = np.zeros((T, 4))
I_lightR[1.2 < t] = np.array([0, 0, 1, 1])
#I_lightR[np.logical_and(1.2 < t, t < 1.5)] = np.array([0, 0, 1, 1])
I_lightR = I_lightR[:,None,None,:,None]
I_lightR = E_light * tf.constant(I_lightR, dtype=DTYPE)

I_LP = I_constant + I_Pbias + I_Prule + I_choice + I_lightL
I_LA = I_constant + I_Pbias + I_Arule + I_choice + I_lightL

I = tf.concat((I_LP, I_LA, I_LP, I_LA), axis=2)

eta = np.ones((T, 1, C, 1, 1), dtype=np.float32)
eta[np.logical_and(0.8 <= t, t <= 1.2), :, 2:, :, :] = opto_strength

def SC_acc(sW_P, sW_A, vW_PA, vW_AP, dW_PA, dW_AP, hW_P, hW_A):
    Wrow1 = tf.stack([sW_P, vW_PA, dW_PA, hW_P], axis=2)
    Wrow2 = tf.stack([vW_AP, sW_A, hW_A, dW_AP], axis=2)
    Wrow3 = tf.stack([dW_AP, hW_A, sW_A, vW_AP], axis=2)
    Wrow4 = tf.stack([hW_P, dW_PA, vW_PA, sW_P], axis=2)
    
    W = tf.stack([Wrow1, Wrow2, Wrow3, Wrow4], axis=2)
    
    # initial conditions
    # M,C,4,N
    state_shape = (sW_P.shape[0], C, 4, N)
    v0 = 0.1 * tf.ones(state_shape, dtype=DTYPE)
    v0 = v0 + 0.005*tf.random.normal(v0.shape, 0., 1.)
    u0 = beta * tf.math.atanh(2 * v0 - 1) - theta

    v = v0
    u = u0
    for i in range(1, T):
        du = (dt / tau) * (-u + tf.matmul(W, v) + I[i] + sigma * tf.random.normal(state_shape, 0., 1.))
        u = u + du
        #v = 1. * (0.5 * tf.tanh((u - theta) / beta) + 0.5)
        v = eta[i] * (0.5 * tf.tanh((u - theta) / beta) + 0.5)

    p = tf.reduce_mean(tf.math.sigmoid(100.*(v[:,:,0,:]-v[:,:,3,:])), axis=2)
    return p

def SC_acc_diff(sW_P, sW_A, vW_PA, vW_AP, dW_PA, dW_AP, hW_P, hW_A):
    p = SC_acc(sW_P, sW_A, vW_PA, vW_AP, dW_PA, dW_AP, hW_P, hW_A)
    p_diffs = p[:,:2] - p[:,2:]
    return p_diffs

model.set_eps(SC_acc_diff)

#init_type = "gaussian"
#init_params = {"mu": np.zeros((model.D,)), "Sigma": (sigma_init**2)*np.eye(model.D)}

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
    K=15,
    N=M,
    num_iters=1000,
    lr=1e-3,
    c0=c0,
    beta=AL_beta,
    nu=0.5,
    #init_type=init_type,
    #init_params=init_params,
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
