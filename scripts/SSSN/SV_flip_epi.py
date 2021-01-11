import os 
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.example_eps import load_W


# Parse script command-line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--logc0', type=float, default=0.) # log10 of c_0
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

c0 = 10.**args.logc0
random_seed = args.random_seed

M = 200

# 1. Specify the V1 model for EPI.
lb_h = 0.*np.ones((4,))
ub_h = 25.*np.array([1., 1., 1., 1.])

lb_dh = -10.
ub_dh = 10.

h = Parameter("h", 4, lb=lb_h, ub=ub_h)
dh = Parameter("dh", 1, lb=lb_dh, ub=ub_dh)
parameters = [h, dh]


# Define model
name = "SSSN_SVflip"
parameters = [h, dh]
model = Model(name, parameters)

V_INIT = tf.constant(-65.*np.ones((1,4,1)), dtype=np.float32)

k = 0.3
n = 2.
v_rest = -70.

dt = 0.005

N = 5
T = 100

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

def f_r(v):
    return k*(tf.nn.relu(v-v_rest)**n)

def SSSN_sim(h):
    h = h[:,None,:,None]

    W = load_W()
    sigma_eps = .2*np.array([1., 0.5, 0.5, 0.5])
    tau = np.array([0.02, 0.01, 0.01, 0.01])
    tau_noise = np.array([0.05, 0.05, 0.05, 0.05])

    W = W[None,:,:,:]
    sigma_eps = sigma_eps[None,None,:,None]
    tau = tau[None,None,:,None]
    tau_noise = tau_noise[None,None,:,None]

    _v_shape = tf.ones((h.shape[0], N, 4, 1), dtype=tf.float32)
    v_init = _v_shape*V_INIT
    eps_init = 0.*_v_shape
    y_init = tf.concat((v_init, eps_init), axis=2)

    def f(y):
        v = y[:,:,:4,:]
        eps = y[:,:,4:,:]
        B = tf.random.normal(eps.shape, 0., np.sqrt(dt))

        dv = (-v + v_rest + h + eps + tf.matmul(W, f_r(v))) / tau
        deps = (-eps + (np.sqrt(2.*tau_noise)*sigma_eps*B/dt)) / tau_noise

        return tf.concat((dv, deps), axis=2)

    v_ss = euler_sim_stoch(f, y_init, dt, T)
    return v_ss

diff_prod_mean = -.25
diff_sum_mean = 0.
def SV_flip(h, dh):
    dh_pattern = tf.constant(np.array([[1., 1., 0., 0.]], dtype=np.float32))
    dh = dh*dh_pattern
   
    ss1 = tf.reduce_mean(f_r(SSSN_sim(h)[:,:,:4]), axis=1)
    ss2 = tf.reduce_mean(f_r(SSSN_sim(h+dh)[:,:,:4]), axis=1)
    
    diff1 = (ss1[:,2]-ss1[:,3]) / tf.norm(ss1, axis=1, keepdims=False)
    diff2 = (ss2[:,2]-ss2[:,3]) / tf.norm(ss2, axis=1, keepdims=False)
    diff_prod = diff1*diff2
    diff_sum = diff1+diff2
    T_x = tf.stack((diff_prod, 
                    diff_sum, 
                    (diff_prod - diff_prod_mean) ** 2, 
                    (diff_sum-diff_sum_mean) ** 2), axis=1)

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
    K=2,
    N=M,
    num_iters=1000,
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
