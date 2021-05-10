"""SMC: RNN stable amplification. """

from neural_circuits.LRRNN import get_W_eigs_np
import numpy as np
import os
import pickle
import tempfile
import pandas as pd
import time

# import matplotlib.pyplot as plt
import argparse

import pyabc
import scipy.stats as st

DTYPE = np.float32

# Get random seed.
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int)
parser.add_argument("--g", type=float, default=0.01)
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--eps", type=float, default=0.5)
parser.add_argument("--rs", type=int, default=1)
args = parser.parse_args()

N = args.N
g = args.g
K = args.K
eps = args.eps
rs = args.rs

sleep_time = N * 0.5 + rs * 0.05
time.sleep(sleep_time)

print("Running SNPE on RNN conditioned on stable amplification with:")
print("N = %d, seed=%d" % (N, rs))

base_path = os.path.join("data", "smc")
save_dir = "SMC_RNN_stab_amp_N=%d_eps=%.2f_rs=%d" % (N, eps, rs)

save_path = os.path.join(base_path, save_dir)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if os.path.exists(os.path.join(base_path, save_dir, "optim.pkl")):
    print("SMC optimization already run. Exitting.")
    exit()

W_eigs = get_W_eigs_np(g, K)


def model(parameter):
    u1 = np.array([parameter["U%d1" % i] for i in range(1, N + 1)])
    u2 = np.array([parameter["U%d2" % i] for i in range(1, N + 1)])
    v1 = np.array([parameter["V%d1" % i] for i in range(1, N + 1)])
    v2 = np.array([parameter["V%d2" % i] for i in range(1, N + 1)])

    U = np.stack((u1, u2), axis=1)
    V = np.stack((v1, v2), axis=1)

    x = W_eigs(U, V)

    return {"data": x}


parameters = [("U%d1" % i, pyabc.RV("uniform", -1.0, 1.0)) for i in range(1, N + 1)]
parameters += [("U%d2" % i, pyabc.RV("uniform", -1.0, 1.0)) for i in range(1, N + 1)]
parameters += [("V%d1" % i, pyabc.RV("uniform", -1.0, 1.0)) for i in range(1, N + 1)]
parameters += [("V%d2" % i, pyabc.RV("uniform", -1.0, 1.0)) for i in range(1, N + 1)]
parameters = dict(parameters)

prior = pyabc.Distribution(parameters)


def distance(x, y):
    return np.linalg.norm(x["data"] - y["data"])


abc = pyabc.ABCSMC(model, prior, distance, population_size=1000)

db_path = "sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db")
observation = np.array([0.5, 1.5])
abc.new(db_path, {"data": observation})

np.random.seed(rs)
max_t = 200
min_acc = 1.0 / (1e6)
time1 = time.time()
history = abc.run(
    minimum_epsilon=eps, max_nr_populations=max_t, min_acceptance_rate=min_acc
)
time2 = time.time()

print("SMC took %.2E hr." % ((time2 - time1) / 3600.0))
df1, w = history.get_distribution(m=0, t=history.max_t)
z = df1.to_numpy()

df = history.get_all_populations()
min_eps = df["epsilon"].min()
converged = min_eps < eps
total_sims = history.total_nr_simulations

optim = {
    "history": history,
    "z": z,
    "eps": eps,
    "time": (time2 - time1),
    "max_t": max_t,
    "min_acc": min_acc,
    "converged": converged,
    "total_sims": total_sims,
}

print("Saving", save_path, "...")
with open(os.path.join(base_path, save_dir, "optim.pkl"), "wb") as f:
    pickle.dump(optim, f)
print("done.")
