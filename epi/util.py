""" General util functions for EPI. """

import numpy as np
import pickle
import os

def gaussian_backward_mapping(mu, Sigma):
    if (len(mu.shape) == 1):
        mu = np.expand_dims(mu, 1)
    D = mu.shape[0]
    Sigma_inv = np.linalg.inv(Sigma)
    x = np.dot(Sigma_inv, mu)
    y = np.reshape(-0.5*Sigma_inv, (D**2))
    eta = np.concatenate((x[:,0], y), 
                         axis=0)
    return eta

def save_tf_model(path, variables):
    d = {}
    for variable in variables:
        d[variable.name] = variable.numpy()
    return pickle.dump(d, open(path+'.p', "wb"))

def load_tf_model(path, variables):
    d = pickle.load(open(path+'.p', "rb"))
    for variable in variables:
        variable.assign(d[variable.name])
    return None

def init_path(arch_string, init_type, init_param):
    path = './data/' + arch_string + '/'
    if (not os.path.exists(path)):
        os.makedirs(path)
        
    if (init_type == 'iso_gauss'):
        loc = init_param['loc']
        scale = init_param['scale']
        path += init_type + '_loc=%.2E_scale=%.2E' % (loc, scale)
    return path
