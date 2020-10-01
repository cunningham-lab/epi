import os 
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.example_eps import euler_sim, euler_sim_traj


# Parse script command-line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--logc0', type=float, default=0.) # log10 of c_0
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

c0 = 10.**args.logc0
random_seed = args.random_seed

# 1. Specify the V1 model for EPI.
lb_h = 0.*np.ones((4,))
ub_h = 3.*np.ones((4,))

lb_dh = -1.*np.ones((2,))
ub_dh = 1.*np.ones((2,))

h = Parameter("h", 4, lb=lb_h, ub=ub_h)
dh = Parameter("dh", 2, lb=lb_dh, ub=ub_dh)

# Define model
sigma_eps = 0.1
name = "V1Circuit_SVflip_sigeps=%.2f" % sigma_eps
parameters = [h, dh]
model = Model(name, parameters)

X_INIT = tf.constant(np.random.normal(1.0, 0.01, (1, 4, 1)).astype(np.float32))

# Define eps
diff_prod_mean = -0.25
diff_sum_mean = 0.
def SV_flip(h, dh):
    h = h[:, :, None]
    dh = tf.concat((dh, tf.zeros_like(dh, dtype=tf.float32)), axis=1)[:, :, None]

    n = 2.
    dt = 0.005
    T = 100
    tau = 0.02

    _x_shape = tf.ones_like(h, dtype=tf.float32)
    x_init = _x_shape*X_INIT

    npzfile = np.load("data/V1_Zs.npz")
    _W = npzfile["Z_allen_square"][None, :, :]
    _W[:, :, 1:] = -_W[:, :, 1:]
    W = tf.constant(_W, dtype=tf.float32)

    def f1(y):
        omega = tf.random.normal(y.shape, 0., 1.)
        noise = sigma_eps*omega
        return (-y + (tf.nn.relu(tf.matmul(W, y) + h + noise) ** n)) / tau

    def f2(y):
        omega = tf.random.normal(y.shape, 0., 1.)
        noise = sigma_eps*omega
        return (-y + (tf.nn.relu(tf.matmul(W, y) + h + dh + noise) ** n)) / tau

    ss1 = euler_sim(f1, x_init, dt, T)
    ss2 = euler_sim(f2, x_init, dt, T)

    diff1 = ss1[:,2]-ss1[:,3]
    diff2 = ss2[:,2]-ss2[:,3]
    diff_prod = diff1*diff2
    diff_sum = diff1+diff2
    T_x = tf.stack((diff_prod,
                    diff_sum,
                    (diff_prod - diff_prod_mean) ** 2,
                    (diff_sum - diff_sum_mean) ** 2), axis=1)

    return T_x

model.set_eps(SV_flip)

# Emergent property values.
mu = np.array([diff_prod_mean, diff_sum_mean, 0.125**2, 0.125**2])

# 3. Run EPI.
q_theta, opt_data, epi_path, failed = model.epi(
    mu,
    arch_type='coupling',
    num_stages=3,
    num_layers=2,
    num_units=50,
    post_affine=True,
    batch_norm=True,
    K=15,
    N=500,
    num_iters=2500,
    lr=1e-3,
    c0=c0,
    beta=4.,
    nu=0.5,
    random_seed=random_seed,
    verbose=True,
    stop_early=True,
    log_rate=50,
    save_movie_data=True,
)

if not failed:
    print("Making movie.")
    model.epi_opt_movie(epi_path)
    print("done.")
