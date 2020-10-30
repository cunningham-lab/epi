import tensorflow as tf
import numpy as np
import os
from scipy.io import loadmat
#import tensorflow_probability as tfp

def load_SSSN_variable(v, ind=0):
    #npzfile = np.load("data/V1_Zs.npz")
    matfile = loadmat(os.path.join("data", "AgosBiorxiv.mat"))
    _x = matfile[v][ind]
    x = tf.constant(_x, dtype=tf.float32)
    return x

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

W_mat = load_SSSN_variable('W', ind=0)
_X_INIT = 0.25*np.array([1., 1., 1., 1.], dtype=np.float32)[None,None,:,None]
X_INIT = tf.constant(_X_INIT, dtype=np.float32)
#X_INIT = tf.constant(.2*np.ones((1,4,1)), dtype=np.float32)
n = 2.
dt = 0.0005
N = 1
T = 500

sigma_eps = 0.0*np.array([1., 0.5, 0.5, 0.5], np.float32)
sigma_eps = sigma_eps[None,None,:,None]
tau = np.array([0.01, 0.001, 0.001, 0.001], np.float32)
tau = tau[None,None,:,None]
tau_noise = np.array([0.05, 0.05, 0.05, 0.05], np.float32)
tau_noise = tau_noise[None,None,:,None]

# Dim is [M,N,|r|,T]
def SSSN_sim_traj(h):
    h = h[:,None,:,None]
    W = W_mat[None,None,:,:]

    _x_shape = tf.ones((h.shape[0], N, 4, 1), dtype=tf.float32)
    x_init = _x_shape*X_INIT
    eps_init = 0.*_x_shape
    y_init = tf.concat((x_init, eps_init), axis=2)

    def f(y):
        x = y[:,:,:4,:]
        eps = y[:,:,4:,:]
        B = tf.random.normal(eps.shape, 0., np.sqrt(dt))

        dx = (-x + (tf.nn.relu(tf.matmul(W, x) + h + eps)**n)) / tau
        deps = (-eps + (np.sqrt(2.*tau_noise)*sigma_eps*B/dt)) / tau_noise

        return tf.concat((dx, deps), axis=2)

    x_t = euler_sim_stoch_traj(f, y_init, dt, T)
    return x_t

def SSSN_sim(h):
    h = h[:,None,:,None]
    W = W_mat[None,None,:,:]
   
    _x_shape = tf.ones((h.shape[0], N, 4, 1), dtype=tf.float32)
    x_init = _x_shape*X_INIT
    eps_init = 0.*_x_shape
    y_init = tf.concat((x_init, eps_init), axis=2)
    
    def f(y):
        x = y[:,:,:4,:]
        eps = y[:,:,4:,:]
        B = tf.random.normal(eps.shape, 0., np.sqrt(dt))

        dx = (-x + (tf.nn.relu(tf.matmul(W, x) + h + eps)**n)) / tau
        deps = (-eps + (np.sqrt(2.*tau_noise)*sigma_eps*B/dt)) / tau_noise
        
        return tf.concat((dx, deps), axis=2)
        
    x_ss = euler_sim_stoch(f, y_init, dt, T)
    return x_ss

"""def SSSN_sim_tfp(h):
    h = h[:,None,:,None]

    W = W_mat[None,None,:,:]

    _x_shape = tf.ones((h.shape[0], N, 4, 1), dtype=tf.float32)
    x_init = _x_shape*X_INIT

    def f(t, x, h):
        dx = (-x + (tf.nn.relu(tf.matmul(W, x) + h)**n)) / tau
        return dx

    results = tfp.math.ode.BDF().solve(f, 0., x_init, 
                                       solution_times=[.25], 
                                       constants={'h': h})
    x_ss = results.states[0]
    return x_ss[:,:,:,0]"""

