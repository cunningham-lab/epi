"""SNPE: RNN stable amplification. """

import numpy as np
import os
import pickle
import tempfile
import pandas as pd
import time
#import matplotlib.pyplot as plt
import argparse

import pyabc
import scipy.stats as st

DTYPE = np.float32

# Get random seed.
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int)
parser.add_argument('--rs', type=int, default=1)
args = parser.parse_args()

N = args.N
rs = args.rs

sleep_time = N*0.5 + rs*0.05
time.sleep(sleep_time)

print('Running SNPE on RNN conditioned on stable amplification with:')
print('N = %d, seed=%d' % (N, rs))

def model(parameter):
    u1 = np.array([parameter["U%d1" % i] for i in range(1, N+1)])
    u2 = np.array([parameter["U%d2" % i] for i in range(1, N+1)])
    v1 = np.array([parameter["V%d1" % i] for i in range(1, N+1)])
    v2 = np.array([parameter["V%d2" % i] for i in range(1, N+1)])

    U = np.stack((u1, u2), axis=1)
    V = np.stack((v1, v2), axis=1)

    J = np.matmul(U, np.transpose(V))
    Js = (J + np.transpose(J)) / 2.
    Js_eigs = np.linalg.eigvalsh(Js)
    Js_eig_max = np.max(Js_eigs, axis=0)

    # Take eig of low rank similar mat
    Jr = np.matmul(np.transpose(V), U) + 0.0001*np.eye(2)
    Jr_tr = np.trace(Jr)
    sqrt_term = np.square(Jr_tr) + -4.*np.linalg.det(Jr)
    if sqrt_term < 0.:
        sqrt_term = 0.
    J_eig_realmax = 0.5 * (Jr_tr + np.sqrt(sqrt_term))
    
    return {"data": np.array([J_eig_realmax, Js_eig_max])}

parameters = [("U%d1" % i,  pyabc.RV("uniform", -1., 1.)) for i in range(1, N+1)]
parameters += [("U%d2" % i, pyabc.RV("uniform", -1., 1.)) for i in range(1, N+1)]
parameters += [("V%d1" % i, pyabc.RV("uniform", -1., 1.)) for i in range(1, N+1)]
parameters += [("V%d2" % i, pyabc.RV("uniform", -1., 1.)) for i in range(1, N+1)]
parameters = dict(parameters)

prior = pyabc.Distribution(parameters)

def distance(x, y):
    return np.linalg.norm(x["data"] - y["data"])

sampler = pyabc.sampler.SingleCoreSampler()

abc = pyabc.ABCSMC(model, prior, distance, sampler=sampler)

db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))
observation = np.array([0.5, 1.5])
abc.new(db_path, {"data": observation})

np.random.seed(rs)
eps = 0.5
max_t = 100
min_acc = 1./(1e7)
time1 = time.time()
history = abc.run(
    minimum_epsilon=eps, 
    max_nr_populations=max_t,
    min_acceptance_rate=min_acc
)
time2 = time.time()
df1, w = history.get_distribution(m=0,t=history.max_t)
z = df1.to_numpy()

df = history.get_all_populations()
min_eps = df['epsilon'].min()
converged = min_eps < eps

optim = {'history':history,
        'z':z,
        'eps':eps,
        'time':(time2-time1),
        'max_t':max_t,
        'min_acc':min_acc,
        'converged':converged,
        }

base_path = os.path.join("data", "smc")
save_dir = "SMC_RNN_stab_amp_N=%d_rs=%d" % (N, rs)

save_path = os.path.join(base_path, save_dir)
if not os.path.exists(save_path):
    os.makedirs(save_path)

print('Saving', save_path, '...')
with open(os.path.join(base_path, save_dir, "optim.pkl"), "wb") as f:
    pickle.dump(optim, f)
print('done.')
