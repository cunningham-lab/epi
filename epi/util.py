""" General util functions for EPI. """

import numpy as np

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
