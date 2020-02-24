"""Run EPI on oscillating 2D LDS. """

from epi.models import Model, Parameter
from epi.example_eps import linear2D_freq
from epi.util import sample_aug_lag_hps
import numpy as np
import tensorflow as tf
import argparse

DTYPE = np.float32

# Get random seed.
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('--d', type=int)
args = parser.parse_args()

print('Running epi for matrix determinant with hyper parameter random seed %d.' % args.seed)
d = args.d


# 1. Define model: dxd matrix
D = d**2

# Set up the bound vectors.
lb_val = -2.
ub_val = 2.
lb_diag = 1.
def off_diag(d, val):
    x = val*np.ones((d,d), dtype=DTYPE)
    for i in range(d):
        x[i,i] = 0.
    return x
lb = np.reshape(lb_diag*np.eye(d) + off_diag(d, lb_val), (D,))
ub = ub_val*np.ones((D,))

# Define the parameter A.
A = Parameter("A", D, lb=lb, ub=ub)
parameters = [A]

# Define the model matrix.
M = Model("matrix", parameters)

# 2. Define the emergent property: E[det(A)] = 100, std(det(A)) = 5
mu = np.array([100., 5.], dtype=np.float32)

@tf.function
def _det(A):
    N = A.get_shape()[0]
    A = tf.reshape(A, (N, d, d))
    detA = tf.linalg.det(A)
    T_x = tf.stack([detA, tf.square(detA-mu[0])], axis=1)
    return T_x

def det(A):
    return _det(A)

M.set_eps(det)


np.random.seed(args.seed)
num_stages = np.random.randint(2, 6) 
num_layers = 2 #np.random.randint(1, 3)
num_units = np.random.randint(15, max(30, D))

init_params = {'loc':0., 'scale':5.}
q_theta, opt_data, save_path = M.epi(
    mu, 
    arch_type='coupling', 
    num_stages=num_stages,
    num_layers=num_layers,
    num_units=num_units,
    post_affine=False,
    batch_norm=False,
    init_params=init_params,
    K=10, 
    num_iters=2500, 
    N=1000,
    lr=1e-3, 
    c0=1e-1, 
    beta=10.,
    verbose=True,
    stop_early=True,
    log_rate=50,
    save_movie_data=True,
)
print("EPI done.")
print("Saved to %s." % save_path)
