import tensorflow as tf
import numpy as np
import scipy
from scipy.special import expit
import matplotlib.pyplot as plt
import os

DTYPE = tf.float32

t_cue_delay = 1.2
t_choice = 0.3
t_post_choice = 0.3
t_total = t_cue_delay + t_choice + t_post_choice
dt = 0.024
t = np.arange(0.0, t_total, dt)
T = t.shape[0]

# input parameters
E_constant = 0.75
E_Pbias = 0.5
E_Prule = 0.6
E_Arule = 0.6
E_choice = 0.25
E_light = 0.5

# set constant parameters
C = 2

theta = 0.05
beta = 0.5
tau = 0.09
sigma = 0.2

# inputs
I_constant = E_constant * tf.ones((T, 1, 1, 4, 1), dtype=DTYPE)

I_Pbias = np.zeros((T, 4))
I_Pbias[t < T * dt] = np.array([1, 0, 0, 1])
I_Pbias = I_Pbias[:,None,None,:,None]
I_Pbias = E_Pbias * tf.constant(I_Pbias, dtype=DTYPE)

I_Prule = np.zeros((T, 4))
I_Prule[t < 1.2] = np.array([1, 0, 0, 1])
I_Prule = I_Prule[:,None,None,:,None]
I_Prule = E_Prule * tf.constant(I_Prule, dtype=DTYPE)

I_Arule = np.zeros((T, 4))
I_Arule[t < 1.2] = np.array([0, 1, 1, 0])
I_Arule = I_Arule[:,None,None,:,None]
I_Arule = E_Arule * tf.constant(I_Arule, dtype=DTYPE)

I_choice = np.zeros((T, 4))
I_choice[t > 1.2] = np.array([1, 1, 1, 1])
I_choice = I_choice[:,None,None,:,None]
I_choice = E_choice * tf.constant(I_choice, dtype=DTYPE)

I_lightL = np.zeros((T, 4))
I_lightL[np.logical_and(1.2 < t, t < 1.5)] = np.array([1, 1, 0, 0])
I_lightL = I_lightL[:,None,None,:,None]
I_lightL = E_light * tf.constant(I_lightL, dtype=DTYPE)

I_lightR = np.zeros((T, 4))
I_lightR[np.logical_and(1.2 < t, t < 1.5)] = np.array([0, 0, 1, 1])
I_lightR = I_lightR[:,None,None,:,None]
I_lightR = E_light * tf.constant(I_lightR, dtype=DTYPE)

I_LP = I_constant + I_Pbias + I_Prule + I_choice + I_lightL
I_LA = I_constant + I_Pbias + I_Arule + I_choice + I_lightL

I = tf.concat((I_LP, I_LA), axis=2)

def SC_sim(sW, vW, dW, hW):
    N = 200
    Wrow1 = tf.stack([sW, vW, dW, hW], axis=2)
    Wrow2 = tf.stack([vW, sW, hW, dW], axis=2)
    Wrow3 = tf.stack([dW, hW, sW, vW], axis=2)
    Wrow4 = tf.stack([hW, dW, vW, sW], axis=2)

    W = tf.stack([Wrow1, Wrow2, Wrow3, Wrow4], axis=2)

    # initial conditions
    # M,C,4,1
    state_shape = (sW.shape[0], C, 4, N)
    v0 = 0.1 * tf.ones(state_shape, dtype=DTYPE)
    v0 = v0 + 0.005*tf.random.normal(v0.shape, 0., 1.)
    u0 = beta * tf.math.atanh(2 * v0 - 1) - theta

    v = v0
    u = u0
    v_t_list = [v]
    u_t_list = [u]
    for i in range(1, T):
        du = (dt / tau) * (-u + tf.matmul(W, v) + I[i] + sigma * tf.random.normal(state_shape, 0., 1.))
        #du = (dt / tau) * (-u + tf.matmul(W, v) + I[i] + sigma * w[i])
        u = u + du
        v = 1. * (0.5 * tf.tanh((u - theta) / beta) + 0.5)
        #v = eta[i] * (0.5 * tf.tanh((u - theta) / beta) + 0.5)
        v_t_list.append(v)
        u_t_list.append(u)

    u_t = tf.stack(u_t_list, axis=0)
    v_t = tf.stack(v_t_list, axis=0)
    return u_t, v_t

def unwrap(z):
    sW = z[:,0][:,None]
    vW = z[:,1][:,None]
    dW = z[:,2][:,None]
    hW = z[:,3][:,None]
    return sW, vW, dW, hW


def SC_acc(sW, vW, dW, hW):
    N = 500
    Wrow1 = tf.stack([sW, vW, dW, hW], axis=2)
    Wrow2 = tf.stack([vW, sW, hW, dW], axis=2)
    Wrow3 = tf.stack([dW, hW, sW, vW], axis=2)
    Wrow4 = tf.stack([hW, dW, vW, sW], axis=2)

    W = tf.stack([Wrow1, Wrow2, Wrow3, Wrow4], axis=2)

    # initial conditions
    # M,C,4,N
    state_shape = (sW.shape[0], C, 4, N)
    v0 = 0.1 * tf.ones(state_shape, dtype=DTYPE)
    v0 = v0 + 0.005*tf.random.normal(v0.shape, 0., 1.)
    u0 = beta * tf.math.atanh(2 * v0 - 1) - theta

    v = v0
    u = u0
    for i in range(1, T):
        du = (dt / tau) * (-u + tf.matmul(W, v) + I[i] + sigma * tf.random.normal(state_shape, 0., 1.))
        u = u + du
        v = 1. * (0.5 * tf.tanh((u - theta) / beta) + 0.5)
        #v = eta[i] * (0.5 * tf.tanh((u - theta) / beta) + 0.5)

    p = tf.reduce_mean(tf.math.sigmoid(100.*(v[:,:,0,:]-v[:,:,3,:])), axis=2)
    return p

def SC_acc_var(P):
    not_P = 1.-P
    _mu = np.array([[P, not_P]])

    def _SC_acc_var(sW, vW, dW, hW):
        N = 200
        Wrow1 = tf.stack([sW, vW, dW, hW], axis=2)
        Wrow2 = tf.stack([vW, sW, hW, dW], axis=2)
        Wrow3 = tf.stack([dW, hW, sW, vW], axis=2)
        Wrow4 = tf.stack([hW, dW, vW, sW], axis=2)
        W = tf.stack([Wrow1, Wrow2, Wrow3, Wrow4], axis=2)

        # initial conditions
        # M,C,4,N
        state_shape = (sW.shape[0], C, 4, N)
        v0 = 0.1 * tf.ones(state_shape, dtype=DTYPE)
        v0 = v0 + 0.005*tf.random.normal(v0.shape, 0., 1.)
        u0 = beta * tf.math.atanh(2 * v0 - 1) - theta

        v = v0
        u = u0
        for i in range(1, T):
            du = (dt / tau) * (-u + tf.matmul(W, v) + I[i] + sigma * tf.random.normal(state_shape, 0., 1.))
            u = u + du
            v = 1. * (0.5 * tf.tanh((u - theta) / beta) + 0.5)

        p = tf.reduce_mean(tf.math.sigmoid(100.*(v[:,:,0,:]-v[:,:,3,:])), axis=2)
        p_var = (p-_mu)**2
        T_x = tf.concat((p, p_var), axis=1)

        return T_x

    return _SC_acc_var

C_opto = 4
I_opto = tf.concat((I_LP, I_LA, I_LP, I_LA), axis=2)

def SC_sim_opto(strength, period):
    print('str', strength)
    eta = np.ones((T, 1, C_opto, 1, 1), dtype=np.float32)
    if period == 'delay':
        eta[np.logical_and(0.8 <= t, t <= 1.2), :, 2:, :, :] = strength
    elif period == 'choice':
        eta[t >= 1.2, :, 2:, :, :] = strength
    elif period == 'total':
        eta[:, :, 2:, :, :] = strength

    def _SC_sim_opto(sW, vW, dW, hW):
        N = 200
        Wrow1 = tf.stack([sW, vW, dW, hW], axis=2)
        Wrow2 = tf.stack([vW, sW, hW, dW], axis=2)
        Wrow3 = tf.stack([dW, hW, sW, vW], axis=2)
        Wrow4 = tf.stack([hW, dW, vW, sW], axis=2)

        W = tf.stack([Wrow1, Wrow2, Wrow3, Wrow4], axis=2)

        # initial conditions
        # M,C,4,1
        state_shape = (sW.shape[0], C_opto, 4, N)
        v0 = 0.1 * tf.ones(state_shape, dtype=DTYPE)
        v0 = v0 + 0.005*tf.random.normal(v0.shape, 0., 1.)
        u0 = beta * tf.math.atanh(2 * v0 - 1) - theta

        v = v0
        u = u0
        v_t_list = [v]
        u_t_list = [u]
        for i in range(1, T):
            du = (dt / tau) * (-u + tf.matmul(W, v) + I_opto[i] + sigma * tf.random.normal(state_shape, 0., 1.))
            u = u + du
            v = eta[i] * (0.5 * tf.tanh((u - theta) / beta) + 0.5)
            v_t_list.append(v)
            u_t_list.append(u)

        u_t = tf.stack(u_t_list, axis=0)
        v_t = tf.stack(v_t_list, axis=0)
        return u_t, v_t
    return _SC_sim_opto

def SC_acc_opto(strength, period):
    eta = np.ones((T, 1, C_opto, 1, 1), dtype=np.float32)
    if period == 'delay':
        eta[np.logical_and(0.8 <= t, t <= 1.2), :, 2:, :, :] = strength
    elif period == 'choice':
        eta[t >= 1.2, :, 2:, :, :] = strength
    elif period == 'total':
        eta[:, :, 2:, :, :] = strength

    def _SC_acc_opto(sW, vW, dW, hW):
        N = 500
        Wrow1 = tf.stack([sW, vW, dW, hW], axis=2)
        Wrow2 = tf.stack([vW, sW, hW, dW], axis=2)
        Wrow3 = tf.stack([dW, hW, sW, vW], axis=2)
        Wrow4 = tf.stack([hW, dW, vW, sW], axis=2)

        W = tf.stack([Wrow1, Wrow2, Wrow3, Wrow4], axis=2)

        # initial conditions
        # M,C,4,N
        state_shape = (sW.shape[0], C_opto, 4, N)
        v0 = 0.1 * tf.ones(state_shape, dtype=DTYPE)
        v0 = v0 + 0.005*tf.random.normal(v0.shape, 0., 1.)
        u0 = beta * tf.math.atanh(2 * v0 - 1) - theta

        v = v0
        u = u0
        for i in range(1, T):
            du = (dt / tau) * (-u + tf.matmul(W, v) + I_opto[i] + sigma * tf.random.normal(state_shape, 0., 1.))
            u = u + du
            v = eta[i] * (0.5 * tf.tanh((u - theta) / beta) + 0.5)

        p = tf.reduce_mean(tf.math.sigmoid(100.*(v[:,:,0,:]-v[:,:,3,:])), axis=2)
        return p

    return _SC_acc_opto

def SC_acc_diff(strength, period):
    sc_acc_opto = SC_acc_opto(strength, period)
    def _SC_acc_diff(sW, vW, dW, hW):
        p = sc_acc_opto(sW, vW, dW, hW)
        p_diffs = p[:,:2] - p[:,2:]
        return p_diffs

    return _SC_acc_diff

def z_to_W(z):
    sW = z[:,0]
    vW = z[:,1]
    dW = z[:,2]
    hW = z[:,3]

    Wrow1 = tf.stack([sW, vW, dW, hW], axis=1)
    Wrow2 = tf.stack([vW, sW, hW, dW], axis=1)
    Wrow3 = tf.stack([dW, hW, sW, vW], axis=1)
    Wrow4 = tf.stack([hW, dW, vW, sW], axis=1)
    W = tf.stack([Wrow1, Wrow2, Wrow3, Wrow4], axis=1)
    return W

def get_schur_eigs(W):
    # returns 
    T, Z = scipy.linalg.schur(W)
    b = Z.copy()
    b[b<0.0] = -1
    b[b>0.0] = 1
    modes = 0.25*np.array([[1.0, 1.0, 1.0, 1.0],   # all mode
                          [-1.0, -1.0, 1.0, 1.0], # side mode
                          [1.0, -1.0, -1.0, 1.0], # task mode
                          [-1.0, 1.0, -1.0, 1.0]]) # diag mode
    X = np.abs(np.dot(modes,b))  # (template_mode x z_col)
    eigs = np.zeros((4,))
    z_inds = []
    for i in range(4):
        z_ind = np.argmax(X[i] == 1.0)
        z_inds.append(z_ind)
        eigs[i] = T[z_ind, z_ind]
    #print(z_inds)
    #print(T)
    return eigs

c_LP = '#3B8023'
c_LA = '#EA8E4C'
c_RA = '#F4C673'
c_RP = '#81C176'

def plot_SC_responses(v_t, fname, figsize=(6,4)):
    M = v_t.shape[1]
    T_x = np.mean(expit(100.*(v_t[-1,:,:,0,:] - v_t[-1,:,:,3,:])), axis=2)
    
    percfont = {'family': 'arial',
                'weight': 'light',
                'size': 20,
            }
    neuron_labels = ['LP', 'LA', 'RA', 'RP']
    colors = [c_LP, c_LA, c_RA, c_RP]
    C_titles = ['Pro, left trials', 'Anti, left trials']
    for m in range(M):
        print("%.1f, %.1f" % (100*T_x[m,0], 100-100*T_x[m,1]))
        fig, axs = plt.subplots(2,1,figsize=figsize)
        for c in range(2):
            for i in range(4):
                mean_v = np.mean(v_t[:,m,c,i,:], axis=1)
                std_v = np.std(v_t[:,m,c,i,:], axis=1)
                axs[c].fill_between(t, mean_v - std_v, mean_v + std_v, color=colors[i], alpha=0.2)
                axs[c].plot(t, mean_v, label=neuron_labels[i], c=colors[i])
                #axs[c].set_title(C_titles[c])
                axs[c].set_ylim([0,1])
                if c == 0:
                    axs[c].text(0.75, 0.5, '%2d%%' % round(100.*T_x[m,c]), fontdict=percfont)
                else:
                    axs[c].text(0.75, 0.5, '%2d%%' % round(100.*(1.-T_x[m,c])), fontdict=percfont)
            if c == 1:
                axs[c].set_xlabel('t (s)', fontsize=18)
        #axs[0].set_ylabel('Activation')
        plt.tight_layout()
        plt.savefig(fname + "sim%d.pdf" % (m+1))
        plt.show()
    return None
