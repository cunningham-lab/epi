"""Run EPI on Rank2 RNN. """

from neural_circuits.LRRNN import get_W_eigs_tf
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
parser.add_argument('--g', type=float, default=0.01)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--c0', type=float)
parser.add_argument('--rs', type=int)
args = parser.parse_args()

print('Running epi for RNN N=%d, c0=%f, rs=%d stable amplification.' % (args.N, args.c0, args.rs))
N = args.N
g = args.g
K = args.K
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

W_eigs = get_W_eigs_tf(g, K)
def stable_amplification_r2(U, V):
    U = tf.reshape(U, (-1, N, 2))
    V = tf.reshape(V, (-1, N, 2))
    T_x = W_eigs(U, V)
    return T_x

M.set_eps(stable_amplification_r2)

q_theta, opt_data, save_path, failed = M.epi(
    mu, 
    batch_norm=False,
    lr=1e-3, 
    N = 100,
    num_iters = 1000,
    c0=c0,
    beta = 4.,
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
