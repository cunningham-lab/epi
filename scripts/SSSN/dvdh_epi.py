import os
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.example_eps import load_W
import time

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

npzfile = np.load('data/SV_mode.npz')
z_mode1 = npzfile['z_mode1'][0]
DH = z_mode1[4]
H = z_mode1[:4] + contrast*DH*np.array([1., 1., 0., 0.])

neuron_inds = {'E':0, 'P':1, 'S':2, 'V':3}
neuron_ind = neuron_inds[alpha]

M = 100

# Dim is [M,N,|r|,T]
def euler_sim_stoch(f, x_init, dt, T):
    x = x_init
    for t in range(T):
        x = x + f(x) * dt
    return x[:, :, :, 0]

def euler_sim_stoch_traj(f, x_init, dt, T):
    x = x_init
    xs = [x_init]
    for t in range(T):
        x = x + f(x) * dt
        xs.append(x)
    return tf.concat(xs, axis=3)

# 1. Specify the V1 model for EPI.
D = 4
lb = -10.*np.ones((D,))
ub = 10.*np.ones((D,))

dh = Parameter("dh", D, lb=lb, ub=ub)

# Define model
name = "SSSN_dvdh_%s_c=%.1f" % (alpha, contrast)
parameters = [dh]
model = Model(name, parameters)

V_INIT = tf.constant(-65.*np.ones((1,4,1)), dtype=np.float32)

k = 0.3
n = 2.
v_rest = -70.

dt = 0.005

N = 1
T = 100

def f_r(v):
    return k*(tf.nn.relu(v-v_rest)**n)

def SSSN_sim(h):
    h = h[:,None,:,None]

    W = load_W()
    sigma_eps = 0.2*np.array([1., 0.5, 0.5, 0.5])
    tau = np.array([0.02, 0.01, 0.01, 0.01])
    tau_noise = np.array([0.05, 0.05, 0.05, 0.05])

    W = W[None,:,:,:]
    sigma_eps = sigma_eps[None,None,:,None]
    tau = tau[None,None,:,None]
    tau_noise = tau_noise[None,None,:,None]

    _v_shape = tf.ones((h.shape[0], N, 4, 1), dtype=tf.float32)
    v_init = _v_shape*V_INIT
    eps_init = 0.*_v_shape
    y_init = tf.concat((v_init, eps_init), axis=2)

    def f(y):
        v = y[:,:,:4,:]
        eps = y[:,:,4:,:]
        B = tf.random.normal(eps.shape, 0., np.sqrt(dt))

        dv = (-v + v_rest + h + eps + tf.matmul(W, f_r(v))) / tau
        deps = (-eps + (np.sqrt(2.*tau_noise)*sigma_eps*B/dt)) / tau_noise

        return tf.concat((dv, deps), axis=2)

    v_ss = euler_sim_stoch(f, y_init, dt, T)
    return v_ss

def dv(dh):
    h = tf.constant(H[None,:], dtype=tf.float32)

    x1 = f_r(SSSN_sim(h)[:,:,neuron_ind])
    x2 = f_r(SSSN_sim(h + dh)[:,:,neuron_ind])

    diff = x2 - x1
    T_x = tf.concat((diff, diff ** 2), axis=1)

    return T_x

model.set_eps(dv)

# Emergent property values.
mu = np.array([0., 1.])

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
