import tensorflow as tf
import numpy as np
import os
from scipy.io import loadmat
#import tensorflow_probability as tfp

def load_SSSN_variable(v, ind=0):
    #npzfile = np.load("data/V1_Zs.npz")
    matfile = loadmat(os.path.join("data", "AgosBiorxiv2.mat"))
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

ind = 1070
W_mat = load_SSSN_variable('W', ind=ind)
HB = load_SSSN_variable('hb', ind=ind)
HC = load_SSSN_variable('hc', ind=ind)
_X_INIT = 0.25*np.array([1., 1., 1., 1.], dtype=np.float32)[None,None,:,None]
X_INIT = tf.constant(_X_INIT, dtype=np.float32)

n = 2.
dt = 0.00025
N = 1
T = 100

tau = 0.001*np.array([1. , 1., 1., 1.], np.float32)
tau = tau[None,None,:,None]
tau_noise = 0.005*np.array([1., 1., 1., 1.], np.float32)
tau_noise = tau_noise[None,None,:,None]

# Dim is [M,N,|r|,T]
def SSSN_sim_traj(eps):
    sigma_eps = eps*np.array([1., 1., 1., 1.], np.float32)
    sigma_eps = sigma_eps[None,None,:,None]
    def _SSSN_sim_traj(h):
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
    return _SSSN_sim_traj

def SSSN_sim(eps, N=1):
    sigma_eps = eps*np.array([1., 1., 1., 1.], np.float32)
    sigma_eps = sigma_eps[None,None,:,None]
    def _SSSN_sim(h):
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

    return _SSSN_sim

def SSSN_sim_sigma_c0(sigma):
    N = 25
    h = HB[None,None,:,None]
    W = W_mat[None,None,:,:]
    sigma = sigma[:,None,:,None]
   
    _x_shape = tf.ones((sigma.shape[0], N, 4, 1), dtype=tf.float32)
    x_init = _x_shape*X_INIT
    eps_init = 0.*_x_shape
    y_init = tf.concat((x_init, eps_init), axis=2)
    
    def f(y):
        x = y[:,:,:4,:]
        eps = y[:,:,4:,:]
        B = tf.random.normal(eps.shape, 0., np.sqrt(dt))

        dx = (-x + (tf.nn.relu(tf.matmul(W, x) + h + eps)**n)) / tau
        deps = (-eps + (np.sqrt(2.*tau_noise)*sigma*B/dt)) / tau_noise
        
        return tf.concat((dx, deps), axis=2)
        
    x_ss = euler_sim_stoch(f, y_init, dt, T)
    return x_ss

def SSSN_sim_sigma_c1(sigma):
    N = 25
    h = (HB+HC)[None,None,:,None]
    W = W_mat[None,None,:,:]
    sigma = sigma[:,None,:,None]
     
    _x_shape = tf.ones((sigma.shape[0], N, 4, 1), dtype=tf.float32)
    x_init = _x_shape*X_INIT
    eps_init = 0.*_x_shape
    y_init = tf.concat((x_init, eps_init), axis=2)
    
    def f(y):
        x = y[:,:,:4,:]
        eps = y[:,:,4:,:]
        B = tf.random.normal(eps.shape, 0., np.sqrt(dt))

        dx = (-x + (tf.nn.relu(tf.matmul(W, x) + h + eps)**n)) / tau
        deps = (-eps + (np.sqrt(2.*tau_noise)*sigma*B/dt)) / tau_noise
        
        return tf.concat((dx, deps), axis=2)
        
    x_ss = euler_sim_stoch(f, y_init, dt, T)
    return x_ss

def ISN_coeff(dh, H):
    sssn_sim = SSSN_sim(0.)
    h = H + dh
    h_E = h[:,0]

    r_ss = sssn_sim(h)
    u_E = tf.linalg.matvec(r_ss[:,0,:4], W_mat[0,:])
    u_E = u_E + h_E
    u_E = tf.nn.relu(u_E)
    isn_coeff = 1.-2.*(u_E)*W_mat[0,0]
    return isn_coeff

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

