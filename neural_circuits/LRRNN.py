import os
import itertools
import pickle
import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from epi.normalizing_flows import NormalizingFlow
from epi.models import Model, Parameter
from epi.util import get_max_H_dist

import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

# EPS = 1e-6


def get_W_eigs_np(g, K, feed_noise=False):
    def W_eigs(U, V, noise=None):
        U, V = U[None, :, :], V[None, :, :]
        U, V = np.tile(U, [K, 1, 1]), np.tile(V, [K, 1, 1])
        if feed_noise:
            U_noise, V_noise = noise
            U = U + g * U_noise
            V = V + g * V_noise
        else:
            U = U + g * np.random.normal(0.0, 1.0, U.shape)
            V = V + g * np.random.normal(0.0, 1.0, V.shape)
        J = np.matmul(U, np.transpose(V, [0, 2, 1]))
        Js = (J + np.transpose(J, [0, 2, 1])) / 2.0
        Js_eigs = np.linalg.eigvalsh(Js)
        Js_eig_maxs = np.max(Js_eigs, axis=1)
        Js_eig_max = np.mean(Js_eig_maxs)

        # Take eig of low rank similar mat
        Jr = np.matmul(np.transpose(V, [0, 2, 1]), U)  # + EPS*np.eye(2)[None,:,:]
        Jr_tr = np.trace(Jr, axis1=1, axis2=2)
        sqrt_term = np.square(Jr_tr) + -4.0 * np.linalg.det(Jr)
        maybe_complex_term = np.sqrt(np.vectorize(complex)(sqrt_term, 0.0))
        J_eig_realmaxs = 0.5 * (Jr_tr + np.real(maybe_complex_term))
        J_eig_realmax = np.mean(J_eig_realmaxs)
        return np.array([J_eig_realmax, Js_eig_max])

    return W_eigs


def get_W_eigs_full_np(g, K, feed_noise=False):
    def W_eigs(U, V, noise=None):
        J = np.matmul(U, np.transpose(V))[None, :, :]
        J = np.tile(J, [K, 1, 1])
        if feed_noise:
            J = J + g * noise
        else:
            J = J + g * np.random.normal(0.0, 1.0, J.shape)
        Js = (J + np.transpose(J, [0, 2, 1])) / 2.0
        Js_eigs = np.linalg.eigvalsh(Js)
        Js_eig_maxs = np.max(Js_eigs, axis=1)
        Js_eig_max = np.mean(Js_eig_maxs)

        J_eig_realmaxs = []
        for k in range(K):
            _J = J[k]
            w, v = np.linalg.eig(_J)
            J_eig_realmaxs.append(np.max(np.real(w)))
        J_eig_realmax = np.mean(J_eig_realmaxs)
        return np.array([J_eig_realmax, Js_eig_max])

    return W_eigs


def get_W_eigs_tf(g, K, Js_eig_max_mean=1.5, J_eig_realmax_mean=0.5, feed_noise=False):
    def W_eigs(U, V, noise=None):
        U, V = U[:, None, :, :], V[:, None, :, :]
        U, V = tf.tile(U, [1, K, 1, 1]), tf.tile(V, [1, K, 1, 1])
        if feed_noise:
            U_noise, V_noise = noise
            U = U + g * U_noise
            V = V + g * V_noise
        else:
            U = U + g * tf.random.normal(U.shape, 0.0, 1.0)
            V = V + g * tf.random.normal(V.shape, 0.0, 1.0)
        J = tf.matmul(U, tf.transpose(V, [0, 1, 3, 2]))
        Js = (J + tf.transpose(J, [0, 1, 3, 2])) / 2.0
        Js_eigs = tf.linalg.eigvalsh(Js)
        Js_eig_maxs = tf.reduce_max(Js_eigs, axis=2)
        Js_eig_max = tf.reduce_mean(Js_eig_maxs, axis=1)

        # Take eig of low rank similar mat
        Jr = tf.matmul(
            tf.transpose(V, [0, 1, 3, 2]), U
        )  # + EPS*tf.eye(2)[None,None,:,:]
        Jr_tr = tf.linalg.trace(Jr)
        maybe_complex_term = tf.sqrt(
            tf.complex(tf.square(Jr_tr) + -4.0 * tf.linalg.det(Jr), 0.0)
        )
        J_eig_realmaxs = 0.5 * (Jr_tr + tf.math.real(maybe_complex_term))
        J_eig_realmax = tf.reduce_mean(J_eig_realmaxs, axis=1)

        T_x = tf.stack(
            [
                J_eig_realmax,
                Js_eig_max,
                tf.square(J_eig_realmax - J_eig_realmax_mean),
                tf.square(Js_eig_max - Js_eig_max_mean),
            ],
            axis=1,
        )
        return T_x

    return W_eigs


def get_W_eigs_full_tf(
    g, K, Js_eig_max_mean=1.5, J_eig_realmax_mean=0.5, feed_noise=False
):
    def W_eigs(U, V, noise=None):
        J = tf.matmul(U, tf.transpose(V, [0, 2, 1]))
        J = tf.tile(J[:, None, :, :], (1, K, 1, 1))
        if feed_noise:
            J = J + g * noise
        else:
            J = J + g * tf.random.normal(J.shape, 0.0, 1.0)
        Js = (J + tf.transpose(J, [0, 1, 3, 2])) / 2.0
        Js_eigs = tf.linalg.eigvalsh(Js)
        Js_eig_maxs = tf.reduce_max(Js_eigs, axis=2)
        Js_eig_max = tf.reduce_mean(Js_eig_maxs, axis=1)

        # Take eig of low rank similar mat
        Jr = tf.matmul(tf.transpose(V, [0, 2, 1]), U)  # + EPS*tf.eye(2)[None,None,:,:]
        Jr_tr = tf.linalg.trace(Jr)
        maybe_complex_term = tf.sqrt(
            tf.complex(tf.square(Jr_tr) + -4.0 * tf.linalg.det(Jr), 0.0)
        )
        J_eig_realmax = 0.5 * (Jr_tr + tf.math.real(maybe_complex_term))
        J_eig_realmax = tf.math.maximum(J_eig_realmax, g)

        T_x = tf.stack(
            [
                J_eig_realmax,
                Js_eig_max,
                tf.square(J_eig_realmax - J_eig_realmax_mean),
                tf.square(Js_eig_max - Js_eig_max_mean),
            ],
            axis=1,
        )
        return T_x

    return W_eigs


RANK = 2


def LRRNN_setup(N, g, K):
    D = int(N * RANK)
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
    return model


def get_epi_optim(model, dist, epi_df, path):
    epi_df = epi_df[epi_df["path"] == path]
    if os.path.isdir(path):
        file = os.path.join(path, "timing.npz")
        try:
            time_file = np.load(file)
            time_per_it = time_file["time_per_it"]
        except:
            print("Error: no timing file %s." % file)
            time_per_it = np.nan
    else:
        print("Error: path not found.")
        return None

    epi_optim = {
        "model": model,
        "dist": dist,
        "iteration": epi_df["iteration"].to_numpy(),
        "H": epi_df["H"].to_numpy(),
        "R1": epi_df["R1"].to_numpy(),
        "R2": epi_df["R2"].to_numpy(),
        "R3": epi_df["R3"].to_numpy(),
        "R4": epi_df["R4"].to_numpy(),
        "time_per_it": time_per_it,
    }
    return epi_optim


def load_ME_EPI_LRRNN(
    N, g, K, mu, random_seeds=None, return_df=False, by_df=False, nu=1.0
):
    D = int(N * RANK)
    model = LRRNN_setup(N, g, K)
    # Choose max entropy
    epi_df = model.get_epi_df()
    epi_df["arch_D"] = [row["arch"]["D"] for i, row in epi_df.iterrows()]
    epi_df["c0"] = [row["AL_hps"]["c0"] for i, row in epi_df.iterrows()]
    epi_df["rs"] = [row["arch"]["random_seed"] for i, row in epi_df.iterrows()]
    epi_df = epi_df[(epi_df["arch_D"] == 2 * D) & (epi_df["c0"] == 1000.0)]
    if random_seeds is not None:
        epi_df = epi_df[epi_df["rs"].isin(random_seeds)]
    tf.random.set_seed(0)
    np.random.seed(0)
    dist, path, best_k = get_max_H_dist(
        model, epi_df, mu, alpha=0.05, nu=nu, check_last_k=1, by_df=by_df
    )
    if path is None:
        epi_optim = None
    else:
        epi_optim = get_epi_optim(model, dist, epi_df, path)

    if return_df:
        return epi_optim, epi_df
    else:
        return epi_optim


def load_best_SNPE_LRRNN(
    N, g, K, x0, num_sims=1000, num_batch=200, num_atoms=100, random_seeds=[1, 2, 3]
):
    snpe_base_path = os.path.join("data", "snpe")
    num_transforms = 3
    # Choose best SNPE
    best_val_prob = None
    snpe_optim = None
    for _i, rs in enumerate(random_seeds):
        print("Processing SNPE N=%d, g =%.2f, rs=%d.\r" % (N, g, rs), end="")
        save_dir = (
            "SNPE_RNN_stab_amp_N=%d_sims=%d_batch=%d_transforms=%d_atoms=%d_g=%.4f_K=%d_rs=%d"
            % (N, num_sims, num_batch, num_transforms, num_atoms, g, K, rs)
        )
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

        best_val_prob_i = np.max(optim["round_val_log_probs"])
        if best_val_prob is None or best_val_prob_i > best_val_prob:
            best_val_prob = best_val_prob_i
            snpe_optim = optim
    print("\n", end="")
    return snpe_optim


def get_simulator(N, g, K):
    _W_eigs = get_W_eigs_np(g, K)

    num_dim = 2 * N * RANK
    prior = utils.BoxUniform(
        low=-1.0 * torch.ones(num_dim), high=1.0 * torch.ones(num_dim)
    )

    def simulator(params):
        params = params.numpy()
        U = np.reshape(params[: (RANK * N)], (N, RANK))
        V = np.reshape(params[(RANK * N) :], (N, RANK))
        x = _W_eigs(U, V)
        return x

    return simulator, prior


def get_epi_times(optim):
    iteration = optim["iteration"]
    return iteration * optim["time_per_it"]


def get_snpe_times(optim, num_sims=None):
    epochs_per_round = np.array(optim["summary"]["epochs"])
    num_rounds = len(epochs_per_round)
    # time of sampling period in round
    sample_times = np.array(optim["sample_times"])
    # time of optimization period in round
    opt_times = np.array(optim["opt_times"])
    distances = np.array(optim["distances"])

    # calc times
    round_times = np.cumsum(sample_times + opt_times)
    epoch_times = [
        num_epochs * [opt_time / num_epochs]
        for num_epochs, opt_time in zip(epochs_per_round, opt_times)
    ]
    for i, _epoch_times in enumerate(epoch_times):
        _epoch_times[0] += sample_times[i]
    epoch_times = np.cumsum(list(itertools.chain.from_iterable(epoch_times)))

    epoch_times = np.concatenate((np.array([0.0]), epoch_times))
    round_times = np.concatenate((np.array([0.0]), round_times))
    if num_sims is not None:
        epoch_sims = [[0]] + [
            num_epochs * [(i + 1) * num_sims] for i, num_epochs in enumerate(epochs_per_round)
        ]
        epoch_sims = list(itertools.chain.from_iterable(epoch_sims))
        round_sims = np.concatenate(
            (np.array([0.0]), np.cumsum(num_rounds * [num_sims])), axis=0
        )

    if num_sims is not None:
        return epoch_times, round_times, epoch_sims, round_sims
    else:
        return epoch_times, round_times

def get_SNPE_conv(N, g, K, x0, eps, 
                        num_sims=1000, 
                        num_batch=200, 
                        num_atoms=100,
                        random_seeds=None):
   
    conv_times = []
    conv_sims = []
    snpe_base_path = os.path.join("data", "snpe")
    num_transforms = 3
    # Choose best SNPE
    for _i, rs in enumerate(random_seeds):
        print("Processing SNPE N=%d, g =%.2f, rs=%d." % (N, g, rs))
        save_dir = (
            "SNPE_RNN_stab_amp_N=%d_sims=%d_batch=%d_transforms=%d_atoms=%d_g=%.4f_K=%d_rs=%d"
            % (N, num_sims, num_batch, num_transforms, num_atoms, g, K, rs)
        )
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
        
        epoch_times, round_times, epoch_sims, round_sims = get_snpe_times(optim, num_sims)
        distances, args = optim['distances'], optim['args']
        num_rounds = len(distances) - 1
        round_val_log_probs = optim['round_val_log_probs']
        if ((num_rounds > 5) and \
               (round_val_log_probs[-3] > round_val_log_probs[-2]) and \
               (round_val_log_probs[-3] > round_val_log_probs[-1])):
            pass
        else:
            print(num_rounds, round_val_log_probs[-4:])
            print('bad')
        
        assert(num_rounds < args.max_rounds)
        conv_inds = np.nonzero(distances<eps)[0]
        conv_time = np.nan if len(conv_inds) == 0 else round_times[conv_inds[0]]
        _conv_sims = np.nan if len(conv_inds) == 0 else round_sims[conv_inds[0]]
        conv_times.append(conv_time)
        conv_sims.append(_conv_sims)
    
    return conv_times, conv_sims


def get_EPI_conv(N, g, K, random_seeds, eps=None):
    D = int(N * RANK)
    model = LRRNN_setup(N, g, K)
    # Choose max entropy
    epi_df = model.get_epi_df()
    epi_df["arch_D"] = [row["arch"]["D"] for i, row in epi_df.iterrows()]
    epi_df["c0"] = [row["AL_hps"]["c0"] for i, row in epi_df.iterrows()]
    epi_df["rs"] = [row["arch"]["random_seed"] for i, row in epi_df.iterrows()]
    epi_df = epi_df[(epi_df["arch_D"] == 2 * D) & (epi_df["c0"] == 1000.0)]
    if random_seeds is not None:
        epi_df = epi_df[epi_df["rs"].isin(random_seeds)]
    paths = sorted(epi_df['path'].unique())
    
    conv_times = []
    conv_sims = []
    for path in paths:
        _epi_df = epi_df[epi_df['path']==path]
        epi_optim = get_epi_optim(model, None, _epi_df, path)
        epi_times = get_epi_times(epi_optim)
        if eps is not None:
            R = np.stack((_epi_df['R1'].to_numpy(), _epi_df['R2'].to_numpy()), axis=1)
            distances = np.linalg.norm(R, axis=1)
            conv_inds = np.nonzero(distances<eps)[0]
        else:
            converged = _epi_df['converged'].to_numpy()==1.
            conv_inds = np.nonzero(converged)[0]

        epi_batch = _epi_df.iloc[0]['AL_hps']['N']
        iterations = epi_optim['iteration']
        
        conv_time = np.nan if len(conv_inds) == 0 else epi_times[conv_inds[0]]
        _conv_sims = np.nan if len(conv_inds) == 0 else epi_batch*iterations[conv_inds[0]]
        conv_times.append(conv_time)
        conv_sims.append(_conv_sims)
   
    return conv_times, conv_sims

def get_SMC_conv(N, random_seeds):
    smc_times = []
    smc_sims = []
    for _rs in random_seeds:
        base_path = os.path.join("data", "smc")
        save_dir = "SMC_RNN_stab_amp_N=%d_rs=%d" % (N, _rs)
        save_path = os.path.join(base_path, save_dir)
    
        try:
            with open(os.path.join(save_path, "optim.pkl"), "rb") as f:
                optim = pickle.load(f)
        except:
            #smc_times.append(np.nan)
            #smc_sims.append(np.nan)
            continue
            
        history = optim["history"]
        converged = optim["converged"]

        if converged:
            smc_times.append(optim['time'])
            smc_sims.append(optim['total_sims'])
        else:
            smc_times.append(np.nan)
            smc_sims.append(np.nan)
    return smc_times, smc_sims



def tf_num_params(N):
    D = int(2 * N * RANK)
    nf = NormalizingFlow(
        D=D,
        arch_type="coupling",
        num_stages=3,
        num_layers=2,
        num_units=100,
        elemwise_fn="affine",
        batch_norm=False,
        bn_momentum=0.0,
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
    density_estimator_build_fun = posterior_nn(
        model="maf",
        hidden_features=50,
        num_transforms=3,
        z_score_x=False,
        z_score_theta=False,
        support_map=True,
    )
    theta, x = simulate_for_sbi(
        simulator, proposal=prior, num_simulations=10, show_progress_bar=False
    )
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


def sim_r2RNN(z, N, M, t):
    U = np.reshape(z[:, : (2 * N)], (-1, N, 2))
    V = np.reshape(z[:, (2 * N) :], (-1, N, 2))
    J = np.matmul(U, np.transpose(V, [0, 2, 1]))
    J = J + np.random.normal(0.0, 0.01, J.shape)

    r_ts = []
    for m in range(M):
        _J = J[m] + 0.01 * np.eye(N)
        f = lambda r, t: (1 / 0.1) * (-r + np.dot(_J, r))
        u, s, v = np.linalg.svd(_J, full_matrices=False)
        r_R_t = scipy.integrate.odeint(f, v[0], t)
        r_ts.append(np.linalg.norm(r_R_t, axis=1))
    r_ts = np.array(r_ts)
    return r_ts


def eig_scatter(T_xs, colors, ax=None, perm=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    alpha = 1.0
    grayval = 0.7
    gray = grayval * np.ones(3)

    T_x = np.concatenate(T_xs, axis=0)
    num_samps = [T_x.shape[0] for T_x in T_xs]
    c = np.concatenate(
        [
            np.array(num_samp * [np.array(color)])
            for num_samp, color in zip(num_samps, colors)
        ],
        axis=0,
    )
    if perm:
        total_samps = sum(num_samps)
        perm = np.random.permutation(total_samps)
        T_x, c = T_x[perm], c[perm]
    ax.plot([0.5], [1.5], "*", c=gray, markersize=10)
    ax.scatter(
        T_x[:, 0], T_x[:, 1], c=c, edgecolors="k", linewidth=0.5, s=50, alpha=alpha
    )
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlim([-1, 3])
    ax.set_ylim([0, 4])
    return None

def snpe_simplot(snpes, label, axs, color=None, max_sims=None, max_time=None, max_dist=None):
    for i, snpe_optim in enumerate(snpes):
        num_sims = snpe_optim['args'].num_sims
        epoch_times, round_times, epoch_sims, round_sims = get_snpe_times(snpe_optim, num_sims)
        snpe_distance = snpe_optim['distances']
        _label = label if i==0 else None
        axs[0].plot(round_sims, snpe_distance, color=color, label=_label)
        axs[1].plot(round_times/60., snpe_distance, color=color)
        _max_dist = np.max(snpe_distance)
        if max_dist is None or max_dist < _max_dist:
            max_dist = _max_dist
        _max_time = np.max(round_times)
        if max_time is None or max_time < _max_time:
            max_time = _max_time
        _max_sims = round_sims[-1]
        if max_sims is None or max_sims < _max_sims:
            max_sims = _max_sims
            
    return max_sims, max_time, max_dist
   
def count_str(n):
    if n >= 1e7:
        return "%dm" %  np.round(n/1e6)
    if n >= 1e6:
        return "%.1fm" %  (n/1e6)
    if n >= 1e4:
        return "%dk" %  np.round(n/1e3)
    elif n >= 1e3:
        return "%.1fk" %  (n/1e3)
    else:
        return "%d" % n

def plot_compare_hp(epis, snpes1, snpes2, c_epi, pal, 
               epi_label=None, snpe1_label=None, snpe2_label=None,
               xlims=None,
               ):
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    
    max_sims, max_time, max_dist = snpe_simplot(snpes1, snpe1_label, axs, pal[0])
    max_sims, max_time, max_dist = snpe_simplot(snpes2, snpe2_label, axs, pal[1], \
                                                max_sims, max_time, max_dist)
           
    labeled = False
    for i, epi_dict in enumerate(epis):
        epi_optim = epi_dict['optim']
        epi_df = epi_dict['df']
        if (epi_optim is None):
            continue
        epi_batch = epi_df.iloc[0]['AL_hps']['N']
        iterations = epi_optim['iteration']
        epi_sims = iterations*epi_batch
        epi_R = np.stack((epi_optim['R1'], epi_optim['R2']), axis=1)
        epi_distance = np.linalg.norm(epi_R, axis=1)
        epi_times = iterations*epi_optim['time_per_it']                    
        label = epi_label if not labeled else None
        axs[0].plot(epi_sims, epi_distance, color=c_epi, label=label)
        axs[1].plot(epi_times/60., epi_distance, color=c_epi)
        labeled=True
        
    if xlims is not None:
        if max_sims > xlims[0]:
            axs[0].set_xlim([0, xlims[0]])
    xticks = axs[0].get_xticks()
    xticklabels = [count_str(count) for count in xticks]
    axs[0].set_xticklabels(xticklabels)
    axs[0].set_ylim([0, 1.1*max_dist])
    axs[0].legend(fontsize=12, loc="upper right")

    hrs = 4*np.arange(7)
    xticks = hrs*60
    xticklabels = ["0"] + ["%dhr" % hr for hr in hrs[1:]]
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticklabels)
    if xlims is not None:
        if max_time > xlims[1]:
            axs[1].set_xlim([0, xlims[1]])
    axs[1].set_ylim([0, 1.1*max_dist])

    return fig, axs

