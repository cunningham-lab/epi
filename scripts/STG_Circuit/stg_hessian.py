import os
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
import time
import matplotlib.pyplot as plt
from epi.STG_Circuit import Simulate, Simulate_all, NetworkFreq
from epi.util import pairplot, plot_T_x

# 1. Specify the V1 model for EPI.
D = 2
g_el = Parameter("g_el", 1, lb=4., ub=8.)
g_synA = Parameter("g_synA", 1, lb=0.01, ub=4.)

# Define model
name = "STG"
parameters = [g_el, g_synA]
model = Model(name, parameters)

freq = 0.55

mu = np.array([freq])

def hessian(z, f):
    """Calculates the Hessian.

    :param z: Samples from distribution.
    :type z: np.ndarray
    :param f: Function to compute hessian of
    :type f: func
    :returns: Hessian of log probability with respect to z.
    :rtype: np.ndarray
    """
    z = _set_z_type(z)
    z = tf.Variable(initial_value=z, trainable=True)
    hess_z = _hessian(z, f)
    del z  # Get rid of dummy variable.
    return hess_z.numpy()

def _hessian(z, f):
    with tf.GradientTape(persistent=True) as tape:
        f_z = f(z)
        st = time.time()
        print('calculating the gradient')
        dfdz = tape.gradient(f_z, z)
        print('calculating the gradient done')
        print(time.time() - st)
        print(dfdz)
    return tape.batch_jacobian(dfdz, z)

def _set_z_type(z):
    if type(z) is list:
        z = np.ndarray(z)
    z = z.astype(np.float32)
    return z

dt = 0.025
T = 300
sigma_I = 0.

network_freq = NetworkFreq(dt, T, sigma_I, mu)
Ds = [param.D for param in model.parameters]
def f(z):
    ind = 0
    zs = []
    for D in Ds:
        zs.append(z[:, ind : (ind + D)])
        ind += D
    return network_freq(*zs)

@tf.function
def temp(z):
    T_x = f(z)
    return T_x[:,0]
z_star = np.array([[6.1071167, 1.3093035]], dtype=np.float32)

start_time = time.time()
print('Calculating hessian')
df2dz2 = hessian(z_star, temp)
end_time = time.time()

print(df2dz2)
total_time = end_time-start_time
print('took', total_time, 'seconds')
