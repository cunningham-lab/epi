import tensorflow as tf 
import numpy as np
import os
from scipy.io import loadmat
from epi.util import dbg_check
import matplotlib.pyplot as plt
#import tensorflow_probability as tfp

FANO_EPS = 1e-6
neuron_inds = {'E':0, 'P':1, 'S':2, 'V':3}

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

def tf_ceil(x, max):
    return max-tf.nn.relu(max-x)

def tf_floor(x, min):
    return min+tf.nn.relu(x-min)

def euler_sim_stoch_traj(f, x_init, dt, T):
    x = x_init
    xs = [x_init]
    for t in range(T):
        x = x + f(x) * dt
        xs.append(x)
    return tf.concat(xs, axis=3)

def euler_sim_stoch_traj_bound(f, x_init, dt, T, min=None, max=None):
    x = x_init
    xs = [x_init]
    for t in range(T):
        x = x + f(x) * dt
        if min is not None:
            x = tf_floor(x, min)
        if max is not None:
            x = tf_ceil(x, max)
        xs.append(x)
    return tf.concat(xs, axis=3)

ind = 1070
ind = 1070
W_mat = load_SSSN_variable('W', ind=ind)
HB = load_SSSN_variable('hb', ind=ind)
HC = load_SSSN_variable('hc', ind=ind)

n = 2.
N = 1
#dt = 0.00025
#T = 100

tau = 0.001*np.array([1. , 1., 1., 1.], np.float32)
tau = tau[None,None,:,None]
tau_noise = 0.005*np.array([1., 1., 1., 1.], np.float32)
tau_noise = tau_noise[None,None,:,None]

# Dim is [M,N,|r|,T]
def SSSN_sim_traj(sigma_eps, W_mat, N=1, dt=0.0005, T=150, x_init=None):
    sigma_eps = sigma_eps[:,None,:,None]
    def _SSSN_sim_traj(h, x_init=x_init):
        h = h[:,None,:,None]
        W = W_mat[None,None,:,:]

        _x_shape = tf.ones((h.shape[0], N, 4, 1), dtype=tf.float32)
        if x_init is None:
            x_init = tf.random.uniform((h.shape[0], N, 4, 1), 0.1, 0.25)
        else:
            x_init = x_init[:,:,:,None]
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
        #x_t = euler_sim_stoch_traj_bound(f, y_init, dt, T, None, 1000)
        return x_t
    return _SSSN_sim_traj

def SSSN_sim_traj_sigma(h, W_mat, N=1, dt=0.0005, T=150):
    h = h[:,None,:,None]
    def _SSSN_sim_traj(sigma_eps):
        sigma_eps = sigma_eps[:,None,:,None]
        W = W_mat[None,None,:,:]

        _x_shape = tf.ones((sigma_eps.shape[0], N, 4, 1), dtype=tf.float32)
        x_init = tf.random.uniform((sigma_eps.shape[0], N, 4, 1), 0.1, 0.25)
        eps_init = 0.*_x_shape
        y_init = tf.concat((x_init, eps_init), axis=2)

        def f(y):
            x = y[:,:,:4,:]
            eps = y[:,:,4:,:]
            B = tf.random.normal(eps.shape, 0., np.sqrt(dt))

            dx = (-x + (tf.nn.relu(tf.matmul(W, x) + h + eps)**n)) / tau
            deps = (-eps + (np.sqrt(2.*tau_noise)*sigma_eps*B/dt)) / tau_noise

            return tf.concat((dx, deps), axis=2)

        #x_t = euler_sim_stoch_traj(f, y_init, dt, T)
        x_t = euler_sim_stoch_traj_bound(f, y_init, dt, T, None, 1000)
        return x_t
    return _SSSN_sim_traj

def SSSN_sim(sigma_eps, W_mat, N=1, dt=0.0005, T=150):
    sigma_eps = sigma_eps*np.array([1., 1., 1., 1.], np.float32)
    sigma_eps = sigma_eps[None,None,:,None]
    def _SSSN_sim(h):
        h = h[:,None,:,None]
        W = W_mat[None,None,:,:]
       
        _x_shape = tf.ones((h.shape[0], N, 4, 1), dtype=tf.float32)
        x_init = tf.random.uniform((h.shape[0], N, 4, 1), 0.1, 0.25)
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

def get_drdh(alpha, eps, W_mat, N=1, dt=0.0005, T=150, delta_step=0.01):
    alpha_ind = neuron_inds[alpha]
    sssn_sim = SSSN_sim(eps, W_mat, N=N)
    delta_h = np.zeros((1,4))
    delta_h[0,alpha_ind] = delta_step
    def _drdh(h):
        x1 = tf.reduce_mean(sssn_sim(h)[:,:,alpha_ind], axis=1)
        x2 = tf.reduce_mean(sssn_sim(h+delta_h)[:,:,alpha_ind], axis=1)

        diff = (x2 - x1)/delta_step
        T_x = tf.stack((diff, diff ** 2), axis=1)

        return T_x
    return _drdh

def get_Fano(alpha, sigma_eps, W_mat, N=100, dt=0.0005, T=150, T_ss=100, mu=0.01, k=100.):
    if not (alpha == 'all'):
        alpha_ind = neuron_inds[alpha]

    sssn_sim_traj = SSSN_sim_traj(sigma_eps, W_mat, N=N, dt=dt, T=T)
    def Fano(h):
        if (alpha == 'all'):
            x_t = k*sssn_sim_traj(h)[:,:,:4,T_ss:]
        else:
            x_t = k*sssn_sim_traj(h)[:,:,alpha_ind,T_ss:]
        _means = tf.math.reduce_mean(x_t, axis=-1)
        _vars = tf.square(tf.math.reduce_std(x_t, axis=-1))
        fano = _vars / (_means+FANO_EPS) 
        vars_mean = tf.reduce_mean(fano, axis=1)
        if (alpha == 'all'):
            T_x = tf.concat((vars_mean, tf.square(vars_mean - mu)), axis=1)
        else:
            T_x = tf.stack((vars_mean, tf.square(vars_mean - mu)), axis=1)
        return T_x
    return Fano

def get_Fano_sigma(alpha, W_mat, h, N=100, dt=0.0005, T=150, T_ss=100, mu=0.01, k=100.):
    if not (alpha == 'all'):
        alpha_ind = neuron_inds[alpha]

    sssn_sim_traj = SSSN_sim_traj_sigma(h, W_mat, N=N, dt=dt, T=T)
    def Fano(sigma_eps):
        if (alpha == 'all'):
            x_t = k*sssn_sim_traj(sigma_eps)[:,:,:4,T_ss:]
        else:
            x_t = k*sssn_sim_traj(sigma_eps)[:,:,alpha_ind,T_ss:]
        _means = tf.math.reduce_mean(x_t, axis=-1)
        _vars = tf.square(tf.math.reduce_std(x_t, axis=-1))
        fano = _vars / _means 
        vars_mean = tf.reduce_mean(fano, axis=1)
        if (alpha == 'all'):
            T_x = tf.concat((vars_mean, tf.square(vars_mean - mu)), axis=1)
        else:
            T_x = tf.stack((vars_mean, tf.square(vars_mean - mu)), axis=1)
        return T_x
    return Fano

def plot_contrast_response(c, x, title='', ax=None, linestyle='-', colors=None, fontsize=14):
    if colors is None:
        colors = 4*['k']
    assert(x.shape[0] == c.shape[0])
    if ax is None:
        fig, ax = plt.subplots(1,1)
    for i in range(4):
        ax.plot(100*c, x[:,i], linestyle, c=colors[i], lw=4)
    ax.set_ylim([0., 80])
    ticksize = fontsize-4
    ax.set_xlabel('contrast (%)', fontsize=fontsize)
    ax.set_ylabel('rate (Hz)', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=ticksize)
    plt.setp(ax.get_yticklabels(), fontsize=ticksize)
    return ax

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
