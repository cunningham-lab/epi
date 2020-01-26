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

def get_array_str(a):
    """Derive string from numpy 1-D array using scientific encoding.

    # Arguments
        a (np.array) String is derived from this array.

    # Returns
        array_str (string) Array string
    """

    def repeats_str(num, mult):
        if (mult == 1):
            return "%.2E" % num
        else:
            return "%dx%.2E" % (mult, num)
    
    d = a.shape[0]
    mults = []
    nums = []
    last_num = a[0]
    mult = 1
    for i in range(1, d):
        if a[i] == last_num:
            mult += 1
            if (i == d-1):
                nums.append(last_num)
                mults.append(mult)
        else:
            nums.append(last_num)
            last_num = a[i]
            mults.append(mult)
            mult = 1

  
    array_str = repeats_str(nums[0], mults[0])
    for i in range(1, len(nums)):
        array_str += '_' + repeats_str(nums[i], mults[i])

    return array_str


def init_path(arch_string, init_type, init_param):
    path = './data/' + arch_string + '/'
    if (not os.path.exists(path)):
        os.makedirs(path)
        
    if (init_type == 'iso_gauss'):
        loc = init_param['loc']
        scale = init_param['scale']
        path += init_type + '_loc=%.2E_scale=%.2E' % (loc, scale)

    return path
