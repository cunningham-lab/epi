"""Run EPI on RNNs with various neurons and simultation lengths. """

from epi.models import Model, Parameter
from epi.example_eps import linear2D_freq
from epi.util import sample_aug_lag_hps
import numpy as np
import tensorflow as tf
import argparse

DTYPE = np.float32

# Get random seed.
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int)
parser.add_argument('--T', type=int)
parser.add_argument('--traj', type=int)
args = parser.parse_args()

num_neurons = args.n
T = args.T
if (args.traj==1):
    full_traj = True
elif (args.traj==0):
    full_traj = False
else:
    raise ValueError('--traj must be 0 or 1')

print(num_neurons, T, full_traj)
# 1. Define model: dxd matrix
D = num_neurons**2
lb = -2.*np.ones((D,), np.float32)
ub = 2.*np.ones((D,), np.float32)
J = Parameter("J", D=D, lb=lb, ub=ub)

params = [J]
M = Model('RNN_n=%d_T=%d' % (num_neurons, T), params)

# 2. Define the emergent property:
x0 = tf.constant(np.random.normal(0., 1., (1, num_neurons,1)), dtype=tf.float32)
w = tf.constant(np.random.normal(0., 1., (num_neurons,)), tf.float32)

if (full_traj):
    targ = np.expand_dims(np.sin(4*np.pi*np.arange(T+1)/T), 0)
    def sim(J):
        J_shape = tf.shape(J)
        N = J_shape[0]
        J = tf.reshape(J, (N, num_neurons, num_neurons))
        
        x = tf.tile(x0, (N,1,1))
        xs = [x]
        for t in range(T):
            x = x + tf.tanh(tf.matmul(J, x))
            xs.append(x)
        
        x = tf.concat(xs, axis=2)
        out = tf.reduce_sum(tf.square(tf.tensordot(x, w, [[1], [0]]) - targ), axis=1)
        T_x = tf.stack((out, tf.square(out)), axis=1)
        return T_x

    M.set_eps(sim)
    mu = np.array([0., 0.1], dtype=DTYPE)

else:
    def sim_end(J):
        J_shape = tf.shape(J)
        N = J_shape[0]
        J = tf.reshape(J, (N, num_neurons, num_neurons))

        x = x0
        for t in range(T):
            x = x + tf.tanh(tf.matmul(J, x0))

        out = tf.tensordot(x, w, [[1], [0]])
        T_x = tf.concat((out, tf.square(out)), axis=1)
        return T_x

    M.set_eps(sim_end)
    mu = np.array([0., 0.1], dtype=DTYPE)

np.random.seed(0)
q_theta, opt_data, save_path, flg = M.epi(
    mu, 
    K=1,
    num_iters=10,
    c0=1e-3,
    verbose=True,
)
