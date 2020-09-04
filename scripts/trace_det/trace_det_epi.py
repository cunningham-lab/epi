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
D = int(d*(d+1)/2)

# Set up the bound vectors.
lb = -2.*np.ones((D,))
ub = 2.*np.ones((D,))

# Define the parameter A.
A = Parameter("A", D, lb=lb, ub=ub)
parameters = [A]

# Define the model matrix.
M = Model("matrix", parameters)

# 2. Define the emergent property: E[det(A)] = 100, std(det(A)) = 5
mu = np.array([d, 0., 1., 1.], dtype=DTYPE)

def trace_det(A):
    diag_div = tf.expand_dims(tf.eye(d), 0) + 1.
    A_lower = tfp.math.fill_triangular(A)
    A = (A_lower + tf.transpose(A_lower, [0, 2, 1]))
    e, v = tf.linalg.eigh(A)
    trace = tf.reduce_sum(e, axis=1)
    det = tf.reduce_prod(e, axis=1)
    T_x = tf.stack([trace, det, tf.square(trace-mu[0]), tf.square(det-mu[1])], axis=1)
    return T_x
M.set_eps(trace_det)


np.random.seed(args.seed)
num_stages = 4 #np.random.randint(2, 6) 
num_layers = 2 #np.random.randint(1, 3)
num_units = D #np.random.randint(15, max(30, D))

init_params = {'loc':0., 'scale':1.}
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
    num_iters=5000, 
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
