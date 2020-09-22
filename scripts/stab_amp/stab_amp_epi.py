"""Run EPI on Rank2 RNN. """

from epi.models import Model, Parameter
from epi.example_eps import linear2D_freq
from epi.util import sample_aug_lag_hps
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import argparse

tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

DTYPE = np.float32

# Get random seed.
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int)
parser.add_argument('--c0', type=float)
parser.add_argument('--rs', type=int)
args = parser.parse_args()

print('Running epi for RNN N=%d, c0=%f, rs=%d stable amplification.' % (args.N, args.c0, args.rs))
N = args.N
c0 = args.c0
rs = args.rs

r = 2 # rank-2 networks

num_epochs = 20

# 1. Define model: dxd matrix
D = int(N*r)

# Set up the bound vectors.
lb = -np.ones((D,))
ub = np.ones((D,))

# Define the parameter A.
U = Parameter("U", D, lb=lb, ub=ub)
V = Parameter("V", D, lb=lb, ub=ub)
parameters = [U, V]

# Define the model matrix.
M = Model("Rank2Net", parameters)

# 2. Define the emergent property:
Js_eig_max_mean = 1.5
J_eig_realmax_mean = 0.5
mu = np.array([Js_eig_max_mean, 
               0.25**2, 
               J_eig_realmax_mean,
               0.25**2], dtype=DTYPE)


def stable_amplification_r2(U, V):
    U = tf.reshape(U, (-1, N, 2))
    V = tf.reshape(V, (-1, N, 2))
    J = tf.matmul(U, tf.transpose(V, [0,2,1]))
    Js = (J + tf.transpose(J, [0, 2, 1])) / 2.
    Js_eigs = tf.linalg.eigvalsh(Js)
    Js_eig_max = tf.reduce_max(Js_eigs, axis=1)
    
    # Take eig of low rank similar mat
    Jr = tf.matmul(tf.transpose(V, [0,2,1]), U) + 0.01*tf.eye(2)[None,:,:]
    Jr_tr = tf.linalg.trace(Jr)
    maybe_complex_term = tf.complex(tf.square(Jr_tr) + -4.*tf.linalg.det(Jr), 0.)
    J_eig_realmax = 0.5 * (Jr_tr + tf.math.real(tf.sqrt(maybe_complex_term)))
    
    T_x = tf.stack([Js_eig_max, tf.square(Js_eig_max-Js_eig_max_mean),
                    J_eig_realmax, tf.square(J_eig_realmax-J_eig_realmax_mean)], axis=1)
    return T_x

M.set_eps(stable_amplification_r2)

q_theta, opt_data, save_path, failed = M.epi(
    mu, 
    batch_norm=False,
    lr=1e-3, 
    N = 100,
    c0=c0,
    beta = 10.,
    nu=1.,
    K=num_epochs,
    verbose=True,
    stop_early=True,
    log_rate=50,
    save_movie_data=False,
    random_seed=rs,
)
if not failed:
    print("EPI done.")
    print("Saved to %s." % save_path)
else:
    print("EPI failed.")
