import os
import itertools
import pickle
import numpy as np
import tensorflow as tf
from epi.normalizing_flows import NormalizingFlow
from epi.models import Model, Parameter
from epi.util import get_max_H_dist

import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

#EPS = 1e-6

def get_W_eigs_np(g, K, feed_noise=False):
    def W_eigs(U, V, noise=None):
        U, V = U[None,:,:], V[None,:,:]
        U, V = np.tile(U, [K,1,1]), np.tile(V, [K,1,1])
        if feed_noise:
            U_noise, V_noise = noise
            U = U + g*U_noise
            V = V + g*V_noise
        else:
            U = U + g*np.random.normal(0., 1., U.shape)
            V = V + g*np.random.normal(0., 1., V.shape)
        J = np.matmul(U, np.transpose(V, [0,2,1]))
        Js = (J + np.transpose(J, [0,2,1])) / 2.
        Js_eigs = np.linalg.eigvalsh(Js)
        Js_eig_maxs = np.max(Js_eigs, axis=1)
        Js_eig_max = np.mean(Js_eig_maxs)

        # Take eig of low rank similar mat
        Jr = np.matmul(np.transpose(V,[0,2,1]), U) #+ EPS*np.eye(2)[None,:,:]
        Jr_tr = np.trace(Jr, axis1=1, axis2=2)
        sqrt_term = np.square(Jr_tr) + -4.*np.linalg.det(Jr)
        maybe_complex_term = np.sqrt(np.vectorize(complex)(sqrt_term, 0.))
        J_eig_realmaxs = 0.5 * (Jr_tr + np.real(maybe_complex_term))
        J_eig_realmax = np.mean(J_eig_realmaxs)
        return np.array([J_eig_realmax, Js_eig_max])
    return W_eigs

def get_W_eigs_full_np(g, K, feed_noise=False):
    def W_eigs(U, V, noise=None):
        J = np.matmul(U, np.transpose(V))[None, :, :]
        J = np.tile(J, [K,1,1])
        if feed_noise:
            J = J + g*noise
        else:
            J = J + g*np.random.normal(0., 1., J.shape)
        Js = (J + np.transpose(J, [0,2,1])) / 2.
        Js_eigs = np.linalg.eigvalsh(Js)
        Js_eig_maxs = np.max(Js_eigs, axis=1)
        Js_eig_max = np.mean(Js_eig_maxs)
       
        J_eig_realmaxs = []
        for k in range(K):
            _J = J[k]
            w,v = np.linalg.eig(_J)
            J_eig_realmaxs.append(np.max(np.real(w)))
        J_eig_realmax = np.mean(J_eig_realmaxs)
        return np.array([J_eig_realmax, Js_eig_max])
    return W_eigs

def get_W_eigs_tf(g, K, Js_eig_max_mean=1.5, J_eig_realmax_mean=0.5, feed_noise=False):
    def W_eigs(U, V, noise=None):
        U, V = U[:,None,:,:], V[:,None,:,:]
        U, V = tf.tile(U, [1,K,1,1]), tf.tile(V, [1,K,1,1])
        if feed_noise:
            U_noise, V_noise = noise
            U = U + g*U_noise
            V = V + g*V_noise
        else:
            U = U + g*tf.random.normal(U.shape, 0., 1.)
            V = V + g*tf.random.normal(V.shape, 0., 1.)
        J = tf.matmul(U, tf.transpose(V, [0,1,3,2]))
        Js = (J + tf.transpose(J, [0,1,3,2])) / 2.
        Js_eigs = tf.linalg.eigvalsh(Js)
        Js_eig_maxs = tf.reduce_max(Js_eigs, axis=2)
        Js_eig_max = tf.reduce_mean(Js_eig_maxs, axis=1)

        # Take eig of low rank similar mat
        Jr = tf.matmul(tf.transpose(V, [0,1,3,2]), U) #+ EPS*tf.eye(2)[None,None,:,:]
        Jr_tr = tf.linalg.trace(Jr)
        maybe_complex_term = tf.sqrt(tf.complex(tf.square(Jr_tr) + -4.*tf.linalg.det(Jr), 0.))
        J_eig_realmaxs = 0.5 * (Jr_tr + tf.math.real(maybe_complex_term))
        J_eig_realmax = tf.reduce_mean(J_eig_realmaxs, axis=1)

        T_x = tf.stack([J_eig_realmax, Js_eig_max,
                        tf.square(J_eig_realmax-J_eig_realmax_mean),
                        tf.square(Js_eig_max-Js_eig_max_mean)], axis=1)
        return T_x
    return W_eigs

def get_W_eigs_full_tf(g, K, Js_eig_max_mean=1.5, J_eig_realmax_mean=0.5, feed_noise=False):
    def W_eigs(U, V, noise=None):
        J = tf.matmul(U, tf.transpose(V, [0,2,1]))
        J = tf.tile(J[:,None,:,:], (1,K,1,1))
        if feed_noise:
            J = J + g*noise
        else:
            J = J + g*tf.random.normal(J.shape, 0., 1.)
        Js = (J + tf.transpose(J, [0,1,3,2])) / 2.
        Js_eigs = tf.linalg.eigvalsh(Js)
        Js_eig_maxs = tf.reduce_max(Js_eigs, axis=2)
        Js_eig_max = tf.reduce_mean(Js_eig_maxs, axis=1)

        # Take eig of low rank similar mat
        Jr = tf.matmul(tf.transpose(V, [0,2,1]), U) #+ EPS*tf.eye(2)[None,None,:,:]
        Jr_tr = tf.linalg.trace(Jr)
        maybe_complex_term = tf.sqrt(tf.complex(tf.square(Jr_tr) + -4.*tf.linalg.det(Jr), 0.))
        J_eig_realmax = 0.5 * (Jr_tr + tf.math.real(maybe_complex_term))
        J_eig_realmax = tf.math.maximum(J_eig_realmax, g)

        T_x = tf.stack([J_eig_realmax, Js_eig_max,
                        tf.square(J_eig_realmax-J_eig_realmax_mean),
                        tf.square(Js_eig_max-Js_eig_max_mean)], axis=1)
        return T_x
    return W_eigs

RANK = 2

def load_ME_EPI_LRRNN(N, g, K, mu, return_df=False, by_df=False, nu=1.0):
    # Choose max entropy
    D = int(N*RANK)
    lb = -np.ones((D,))
    ub = np.ones((D,))
    U = Parameter("U", D, lb=lb, ub=ub)
    V = Parameter("V", D, lb=lb, ub=ub)
    parameters = [U, V]
    model = Model("Rank2Net_g=%.4f_K=%d" % (g, K), parameters)
    W_eigs = get_W_eigs_tf(g, K)
    def stable_amp(U, V):
        U = tf.reshape(U, (-1, N, RANK))
        V = tf.reshape(V, (-1, N, RANK))
        T_x = W_eigs(U, V)
        return T_x
    model.set_eps(stable_amp)
    epi_df = model.get_epi_df()
    epi_df['arch_D'] = [row['arch']['D'] for i, row in epi_df.iterrows()]
    epi_df['c0'] = [row['AL_hps']['c0'] for i, row in epi_df.iterrows()]
    epi_df = epi_df[(epi_df['arch_D']==2*D) & (epi_df['c0']==1000.)]
    tf.random.set_seed(0)
    np.random.seed(0)
    dist, path, best_k = get_max_H_dist(model, epi_df, mu, alpha=0.05, nu=nu, 
                                        check_last_k=1, by_df=by_df)
    if path is None:
        if return_df:
            return None, None
        else:
            return None
    epi_df = epi_df[epi_df['path']==path]
    time_filename = os.path.join(path, 'timing.npz')
    time_file = np.load(time_filename)
    time_per_it = time_file['time_per_it']

    epi_optim = {'model':model,
             'dist':dist,
             'iteration':epi_df['iteration'].to_numpy(),
             'H':epi_df['H'].to_numpy(),
             'R1':epi_df['R1'].to_numpy(),
             'R2':epi_df['R2'].to_numpy(),
             'R3':epi_df['R3'].to_numpy(),
             'R4':epi_df['R4'].to_numpy(),
             'time_per_it':time_per_it,}

    if return_df:
        return epi_optim, epi_df
    else:
        return epi_optim

def load_best_SNPE_LRRNN(N, g, K, x0,
                         num_sims=1000, num_batch=200,num_atoms=100, 
                         random_seeds=[1,2,3]):
    snpe_base_path = os.path.join("data", "snpe")
    num_transforms = 3
    # Choose best SNPE
    best_val_prob = None
    snpe_optim = None
    for _i, rs in enumerate(random_seeds):
        print("Processing SNPE N=%d, g =%.2f, rs=%d.\r" % (N,g,rs), end="")
        save_dir = "SNPE_RNN_stab_amp_N=%d_sims=%d_batch=%d_transforms=%d_atoms=%d_g=%.4f_K=%d_rs=%d" \
                % (N, num_sims, num_batch, num_transforms, num_atoms, g, K, rs)
        save_path = os.path.join(snpe_base_path, save_dir)
        if os.path.isdir(save_path):
            file = os.path.join(save_path, "optim.pkl")
            try:
                with open(file, "rb") as f:
                    optim = pickle.load(f)
            except:
                print("Error: no optim file %s." % file)
                continue
        else:
            print("Error: no save path %s." % save_path)
            continue

        best_val_prob_i = np.max(optim['round_val_log_probs'])
        if best_val_prob is None or best_val_prob_i > best_val_prob:
            best_val_prob = best_val_prob_i
            snpe_optim = optim
    print('\n', end="")
    return snpe_optim

def get_simulator(N, g, K):
    _W_eigs = get_W_eigs_np(g, K)

    num_dim = 2*N*RANK
    prior = utils.BoxUniform(low=-1.*torch.ones(num_dim), high=1.*torch.ones(num_dim))

    def simulator(params):
        params = params.numpy()
        U = np.reshape(params[:(RANK*N)], (N,RANK))
        V = np.reshape(params[(RANK*N):], (N,RANK))
        x = _W_eigs(U, V)
        return x
    return simulator, prior

T_EPS = 1e-6
def get_epi_times(optim):
    iteration = optim['iteration']
    return iteration*optim['time_per_it'] + T_EPS

def get_snpe_times(optim, num_sims=None):
    round_durations = np.array(optim['times'])
    round_times = np.cumsum(round_durations)
    summary = optim['summary']
    epochs = np.array(summary['epochs'])
    epoch_durations = round_durations/epochs
    epoch_times = [num_epochs*[epoch_duration] for num_epochs, epoch_duration in zip(epochs, epoch_durations)]
    if num_sims is not None:
        epoch_sims = [[0]] + [num_epochs*[(i+1)*num_sims] for i, num_epochs in enumerate(epochs)]
        epoch_sims = list(itertools.chain.from_iterable(epoch_sims))
        round_sims = np.concatenate(
            (np.array([0.]), np.cumsum(len(round_times)*[num_sims])), 
            axis=0)
    epoch_times = np.cumsum(list(itertools.chain.from_iterable(epoch_times)))
    epoch_times = np.concatenate((np.array([0.]), epoch_times)) + T_EPS
    round_times = np.concatenate((np.array([0.]), round_times)) + T_EPS

    if num_sims is not None:
        return epoch_times, round_times, epoch_sims, round_sims
    else:
        return epoch_times, round_times


def has_converged(z, simulator, x0, eps):
    x = np.array([simulator(torch.tensor(_z)) for _z in z])
    median_x = np.median(x, axis=0)
    distance = np.linalg.norm(median_x - x0)
    return distance < eps

def get_convergence_round(optim, g, K, x0, eps):
    zs = optim['zs']
    num_rounds = zs.shape[0]
    N = zs.shape[2]//4

    simulator, _ = get_simulator(N, g, K)

    _round = None
    if not has_converged(zs[-1], simulator, x0, eps):
        return _round
    for i in range(num_rounds):
        z = zs[i]
        if has_converged(z, simulator, x0, eps):
            _round = i+1
            break
    return _round

def tf_num_params(N):
    D = int(2*N*RANK)
    nf = NormalizingFlow(
        D = D,
        arch_type="coupling",
        num_stages=6,
        num_layers=2,
        num_units=50,
        elemwise_fn="affine",
        batch_norm=False,
        bn_momentum=0.,
        post_affine=True,
        random_seed=1,
    )
    x, log_prob = nf(10)
    num_params = 0
    for tf_var in nf.trainable_variables:
        num_params += np.prod(tf_var.shape)
    return num_params

def torch_num_params(N):
    simulator, prior = get_simulator(N, 0.01, 1)
    simulator, prior = prepare_for_sbi(simulator, prior)
    density_estimator_build_fun = posterior_nn(model='maf', hidden_features=50, num_transforms=3,
                                               z_score_x=False, z_score_theta=False,
                                               support_map=True)
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=10, show_progress_bar=False)
    maf = density_estimator_build_fun(theta, x)
    num_params = 0
    for param in maf.parameters():
        num_params += np.prod(param.shape)
    return num_params

def SNPE_entropy(log_probs):
    snpe_H = []
    for _log_probs in log_probs:
        safe_log_probs = []
        for log_prob in _log_probs:
            if not (np.isnan(log_prob) or np.isinf(log_prob)):
                safe_log_probs.append(log_prob)
        snpe_H.append(-np.mean(safe_log_probs))
    return snpe_H

def eig_scatter(T_xs, colors, ax=None, perm=True):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(5,5))
    alpha = 1.
    grayval = 0.7
    gray = grayval*np.ones(3)

    T_x = np.concatenate(T_xs, axis=0)
    num_samps = [T_x.shape[0] for T_x in T_xs]
    c = np.concatenate([np.array(num_samp*[np.array(color)]) for num_samp, color in zip(num_samps, colors)], axis=0)
    if perm:
        total_samps = sum(num_samps)
        perm = np.random.permutation(total_samps)
        T_x, c = T_x[perm], c[perm]
    ax.plot([0.5], [1.5], '*', c=gray, markersize=10)
    ax.scatter(T_x[:,0], T_x[:,1], c=c,
               edgecolors='k',  linewidth=0.5, s=50,
               alpha=alpha)
    ax.set_yticks([0,1,2,3,4])
    ax.set_xlim([-1, 3])
    ax.set_ylim([0, 4])
    return None
