"""Run EPI on oscillating 2D LDS. """

from epi.models import Model, Parameter
from epi.example_eps import linear2D_freq
from epi.util import sample_aug_lag_hps
import numpy as np
import tensorflow as tf
import argparse

# Get random seed.
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('--d', type=int)
args = parser.parse_args()

print('Running epi for matrix determinant with hyper parameter random seed %d.' % args.seed)
d = args.d

# 1. Define model: dxd matrix
D = d**2
lb = -10.*np.ones(D)
ub = 10.*np.ones(D)
A = Parameter("A", D, lb=lb, ub=ub)
parameters = [A]

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
num_stages = np.random.randint(2, 2*d) 
num_layers = np.random.randint(1, 3)
num_units = np.random.randint(15, max(30, D))

init_params = {'loc':0., 'scale':3.}
q_theta, opt_data, save_path = M.epi(
    mu, 
    arch_type='coupling', 
    num_stages=num_stages,
    num_layers=num_layers,
    num_units=num_units,
    post_affine=False,
    init_params=init_params,
    K=1, 
    num_iters=100, 
    N=500,
    lr=1e-3, 
    c0=1e-3, 
    verbose=True,
    stop_early=True,
    log_rate=50,
    save_movie_data=True,
)
print("EPI done.")
print("Saved to %s." % save_path)
