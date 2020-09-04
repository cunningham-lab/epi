""" Models. """

import numpy as np
import inspect
import tensorflow as tf
from scipy.stats import ttest_ind
from sklearn.neighbors import KernelDensity
from epi.error_formatters import format_type_err_msg
from epi.normalizing_flows import NormalizingFlow
from epi.util import (
    gaussian_backward_mapping,
    aug_lag_vars,
    unbiased_aug_grad,
    AugLagHPs,
    array_str,
    np_column_vec,
    plot_square_mat,
    get_hash,
    set_dir_index,
    get_dir_index,
)
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import seaborn as sns
import pickle
import time
import os

REAL_NUMERIC_TYPES = (int, float)


class Parameter(object):
    """Univariate parameter of a model.

    :param name: Parameter name.
    :type name: str
    :param D: Number of dimensions of parameter.
    :type D: int
    :param lb: Lower bound of variable, defaults to `np.NINF*np.ones(D)`.
    :type lb: np.ndarray, optional
    :param ub: Upper bound of variable, defaults to `np.PINF*np.ones(D)`.
    :type ub: np.ndarray, optional
    """

    def __init__(self, name, D, lb=None, ub=None):
        """Constructor method."""
        self._set_name(name)
        self._set_D(D)
        self._set_bounds(lb, ub)

    def _set_name(self, name):
        if type(name) is not str:
            raise TypeError(format_type_err_msg(self, "name", name, str))
        self.name = name

    def _set_D(self, D):
        if type(D) is not int:
            raise TypeError(format_type_err_msg(self, "D", D, int))
        if D < 1:
            raise ValueError("Dimension of parameter must be positive.")
        self.D = D

    def _set_bounds(self, lb, ub):
        if lb is None:
            lb = np.NINF * np.ones(self.D)
        elif isinstance(lb, REAL_NUMERIC_TYPES):
            lb = np.array([lb])

        if ub is None:
            ub = np.PINF * np.ones(self.D)
        elif isinstance(ub, REAL_NUMERIC_TYPES):
            ub = np.array([ub])

        if type(lb) is not np.ndarray:
            raise TypeError(format_type_err_msg(self, "lb", lb, np.ndarray))
        if type(ub) is not np.ndarray:
            raise TypeError(format_type_err_msg(self, "ub", ub, np.ndarray))

        lb_shape = lb.shape
        if len(lb_shape) != 1:
            raise ValueError("Lower bound lb must be vector.")
        if lb_shape[0] != self.D:
            raise ValueError("Lower bound lb does not have dimension D = %d." % self.D)

        ub_shape = ub.shape
        if len(ub_shape) != 1:
            raise ValueError("Upper bound ub must be vector.")
        if ub_shape[0] != self.D:
            raise ValueError("Upper bound ub does not have dimension D = %d." % self.D)

        for i in range(self.D):
            if lb[i] > ub[i]:
                raise ValueError(
                    "Parameter %s lower bound is greater than upper bound." % self.name
                )
            elif lb[i] == ub[i]:
                raise ValueError(
                    "Parameter %s lower bound is equal to upper bound." % self.name
                )

        self.lb = lb
        self.ub = ub


class Model(object):
    """Model to run emergent property inference on.  To run EPI on a model:

    #. Initialize an :obj:`epi.models.Model` with a list of :obj:`epi.models.Parameter`.
    #. Use :obj:`epi.models.Model.set_eps` to set the emergent property statistics of the model.
    #. Run emergent property inference for mean parameter :math:`\\mu` using :obj:`epi.models.Model.epi`.

    :param name: Name of model.
    :type name: str
    :param parameters: List of :obj:`epi.models.Parameter`.
    :type parameters: list
    """

    def __init__(self, name, parameters):
        self._set_name(name)
        self._set_parameters(parameters)
        self.eps = None
        self.M_test = 200
        self.M_norm = 200

    def _set_name(self, name):
        if type(name) is not str:
            raise TypeError(format_type_err_msg(self, "name", name, str))
        self.name = name

    def _set_parameters(self, parameters):
        if type(parameters) is not list:
            raise TypeError(format_type_err_msg(self, parameters, "parameters", list))
        for parameter in parameters:
            if not parameter.__class__.__name__ == "Parameter":
                raise TypeError(
                    format_type_err_msg(self, "parameter", parameter, Parameter)
                )
        if not self.parameter_check(parameters, verbose=True):
            raise ValueError("Invalid parameter list.")
        self.parameters = parameters
        self.D = sum([param.D for param in parameters])

    def set_eps(self, eps):
        """Set the emergent property statistic calculation for this model.

        The arguments of eps should be batch vectors of univariate parameter
        tensors following the naming convention in :obj:`self.Parameters`.

        :param eps: Emergent property statistics function.
        :type eps: function
        """
        fullargspec = inspect.getfullargspec(eps)
        args = fullargspec.args
        _parameters = []
        Ds = []
        for arg in args:
            found = False
            for param in self.parameters:
                if param.name == arg:
                    found = True
                    _parameters.append(param)
                    Ds.append(param.D)
                    self.parameters.remove(param)
                    break
            if not found:
                raise ValueError(
                    "Function eps has argument %s not in model parameter list." % arg
                )

        self.parameters = _parameters

        def _eps(z):
            ind = 0
            zs = []
            for D in Ds:
                zs.append(z[:, ind : (ind + D)])
                ind += D
            return eps(*zs)

        self.eps = _eps
        self.eps.__name__ = eps.__name__

        # Measure the eps dimensionality to populate self.m.
        z = tf.ones((1, self.D))
        T_z = self.eps(z)
        T_z_shape = T_z.shape
        if len(T_z_shape) != 2:
            raise ValueError("Method eps must return tf.Tensor of dimension (N, D).")
        self.m = T_z_shape[1]
        return None

    def _get_bounds(self,):
        lb = np.zeros((self.D,))
        ub = np.zeros((self.D,))
        ind = 0
        for param in self.parameters:
            lb[ind : (ind + param.D)] = param.lb
            ub[ind : (ind + param.D)] = param.ub
            ind += param.D
        return (lb, ub)

    def epi(
        self,
        mu,
        arch_type="coupling",
        num_stages=3,
        num_layers=2,
        num_units=None,
        batch_norm=True,
        bn_momentum=0.99,
        post_affine=False,
        random_seed=1,
        init_type=None,  # "iso_gauss",
        init_params=None,  # {"loc": 0.0, "scale": 1.0},
        K=10,
        num_iters=1000,
        N=500,
        lr=1e-3,
        c0=1.0,
        gamma=0.25,
        beta=4.0,
        alpha=0.05,
        nu=1.0,
        stop_early=False,
        log_rate=50,
        verbose=False,
        save_movie_data=False,
    ):
        """Runs emergent property inference for this model with mean parameter :math:`\\mu`.


        :param mu: Mean parameter of the emergent property.
        :type mu: np.ndarray
        :param arch_type: :math:`\\in` :obj:`['autoregressive', 'coupling']`, defaults to :obj:`'coupling'`.
        :type arch_type: str, optional
        :param num_stages: Number of coupling or autoregressive stages, defaults to 3.
        :type num_stages: int, optional
        :param num_layers: Number of neural network layer per conditional, defaults to 2.
        :type num_layers: int, optional
        :param num_units: Number of units per layer, defaults to max(2D, 15).
        :type num_units: int, optional
        :param batch_norm: Use batch normalization between stages, defaults to True.
        :type batch_norm: bool, optional
        :param bn_momentum: Batch normalization momentum parameter, defaults to 0.99.
        :type bn_momentrum: float, optional
        :param post_affine: Shift and scale following main transform, defaults to False.
        :type post_affine: bool, optional
        :param random_seed: Random seed of architecture parameters, defaults to 1.
        :type random_seed: int, optional
        :param init_type: :math:`\\in` :obj:`['iso_gauss', 'gaussian']`.
        :type init_type: str, optional
        :param init_params: Parameters according to :obj:`init_type`.
        :type init_params: dict, optional
        :param K: Number of augmented Lagrangian iterations, defaults to 10.
        :type K: int, float, optional
        :param num_iters: Number of optimization iterations, defaults to 1000.
        :type num_iters: int, optional
        :param N: Number of batch samples per iteration, defaults to 500.
        :type N: int, optional
        :param lr: Adam optimizer learning rate, defaults to 1e-3.
        :type lr: float, optional
        :param c0: Initial augmented Lagrangian coefficient, defaults to 1.0.
        :type c0: float, optional
        :param gamma: Augmented lagrangian hyperparameter, defaults to 0.25.
        :type gamma: float, optional
        :param beta: Augmented lagrangian hyperparameter, defaults to 4.0.
        :type beta: float, optional
        :param alpha: P-value threshold for convergence testing, defaults to 0.05.
        :type alpha: float, optional
        :param nu: Fraction of N for convergence testing, defaults to 0.1.
        :type nu: float, optional
        :param stop_early: Exit if converged, defaults to False.
        :type stop_early: bool, optional
        :param log_rate: Record optimization data every so iterations, defaults to 100.
        :type log_rate: int, optional
        :param verbose: Print optimization information, defaults to False.
        :type verbose: bool, optional
        :param save_movie_data: Save data for making optimization movie, defaults to False.
        :type save_movie_data: bool, optional
        :returns: q_theta, opt_df, save_path
        :rtype: epi.models.Distribution, pandas.DataFrame, str
        """
        if num_units is None:
            num_units = max(2 * self.D, 15)

        nf = NormalizingFlow(
            arch_type=arch_type,
            D=self.D,
            num_stages=num_stages,
            num_layers=num_layers,
            num_units=num_units,
            batch_norm=batch_norm,
            bn_momentum=bn_momentum,
            post_affine=post_affine,
            bounds=self._get_bounds(),
            random_seed=random_seed,
        )

        # Hyperparameter object
        aug_lag_hps = AugLagHPs(N, lr, c0, gamma, beta)

        # Initialize architecture to gaussian.
        print("Initializing %s architecture." % nf.to_string(), flush=True)
        if init_params is None:
            mu_init = np.zeros((self.D))
            Sigma = np.zeros((self.D, self.D))
            for i in range(self.D):
                if np.isneginf(nf.lb[i]) and np.isposinf(nf.ub[i]):
                    mu_init[i] = 0.0
                    Sigma[i, i] = 1.0
                elif np.isneginf(nf.lb[i]):
                    mu_init[i] = nf.ub[i] - 2.0
                    Sigma[i, i] = 1.0
                elif np.isposinf(nf.ub[i]):
                    mu_init[i] = nf.lb[i] + 2.0
                    Sigma[i, i] = 1.0
                else:
                    mu_init[i] = (nf.lb[i] + nf.ub[i]) / 2.0
                    Sigma[i, i] = np.square((nf.ub[i] - nf.lb[i]) / 4)
            init_type = "gaussian"
            init_params = {"mu": mu_init, "Sigma": Sigma}
        nf.initialize(init_params["mu"], init_params["Sigma"])

        # Checkpoint the initialization.
        optimizer = tf.keras.optimizers.Adam(lr)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=nf)
        ckpt_dir, exists = self.get_epi_path(init_params, nf, mu, aug_lag_hps)
        if exists:
            print("Loading cached epi at %s." % ckpt_dir)
            q_theta = self._get_epi_dist(-1, init_params, nf, mu, aug_lag_hps)
            opt_df = pd.read_csv(os.path.join(ckpt_dir, "opt_data.csv"), index_col=0)
            failed = (opt_df['cost'].isna()).sum() > 0 
            return q_theta, opt_df, ckpt_dir, failed
        manager = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=None)
        manager.save(checkpoint_number=0)

        print("Saving EPI models to %s." % ckpt_dir, flush=True)

        @tf.function
        def train_step(eta, c):
            with tf.GradientTape(persistent=True) as tape:
                z, log_q_z = nf(N)
                params = nf.trainable_variables
                tape.watch(params)
                H, R, R1s, R2 = aug_lag_vars(z, log_q_z, self.eps, mu, N)
                neg_H = -H
                lagrange_dot = tf.reduce_sum(tf.multiply(eta, R))
            aug_l2 = c / 2.0 * tf.reduce_sum(tf.square(R))
            cost = neg_H + lagrange_dot + aug_l2
            H_grad = tape.gradient(neg_H, params)
            lagrange_grad = tape.gradient(lagrange_dot, params)
            aug_grad = unbiased_aug_grad(R1s, R2, params, tape)
            gradients = [
                g1 + g2 + c * g3 for g1, g2, g3 in zip(H_grad, lagrange_grad, aug_grad)
            ]
            optimizer.apply_gradients(zip(gradients, params))
            return cost, H, R, z, log_q_z

        N_test = int(nu * N)
        # Initialize augmented Lagrangian parameters eta and c.
        eta, c = np.zeros((self.m,), np.float32), c0
        etas, cs = np.zeros((K, self.m)), np.zeros((K,))

        # Initialize optimization data frame.
        z, log_q_z = nf(N)
        H_0, R_0, _, _ = aug_lag_vars(z, log_q_z, self.eps, mu, N)
        cost_0 = -H_0 + np.dot(eta, R_0) + np.sum(np.square(R_0))
        R_keys = ["R%d" % (i + 1) for i in range(self.m)]
        opt_it_dfs = [self._opt_it_df(
            0, 
            0, 
            H_0.numpy(), 
            cost_0.numpy(), 
            R_0.numpy(), 
            log_rate, 
            R_keys)]

        # Record samples for movie.
        if save_movie_data:
            N_save = 200
            zs = [z.numpy()[:N_save, :]]
            log_q_zs = [log_q_z.numpy()[:N_save]]

        # Measure initial R norm distribution.
        mu_colvec = np_column_vec(mu).astype(np.float32).T
        norms = get_R_norm_dist(nf, self.eps, mu_colvec, self.M_norm, N)

        # EPI optimization
        print(format_opt_msg(0, 0, cost_0, H_0, R_0), flush=True)
        failed = False
        for k in range(1, K + 1):
            etas[k - 1], cs[k - 1], eta, c
            for i in range(1, num_iters + 1):
                time1 = time.time()
                cost, H, R, z, log_q_z = train_step(eta, c)
                time2 = time.time()
                if i % log_rate == 0:
                    if verbose:
                        print(format_opt_msg(k, i, cost, H, R), flush=True)
                    it = (k - 1) * num_iters + i
                    opt_it_dfs.append(
                        self._opt_it_df(
                            k, 
                            it, 
                            H.numpy(), 
                            cost.numpy(), 
                            R.numpy(), 
                            log_rate, 
                            R_keys)
                    )
                    if save_movie_data:
                        zs.append(z.numpy()[:N_save, :])
                        log_q_zs.append(log_q_z.numpy()[:N_save])
                if np.isnan(cost):
                    failed = True
                    if verbose:
                        print(format_opt_msg(k, i, cost, H, R), flush=True)
                    it = (k - 1) * num_iters + i
                    opt_it_dfs.append(
                        self._opt_it_df(
                            k, 
                            it, 
                            H.numpy(), 
                            cost.numpy(), 
                            R.numpy(), 
                            log_rate,
                            R_keys)
                    )
                    print("NaN in EPI optimization. Exiting.")
                    break
            if not verbose:
                print(format_opt_msg(k, i, cost, H, R), flush=True)

            # Save epi optimization data following aug lag iteration k.
            opt_it_df = pd.concat(opt_it_dfs)
            manager.save(checkpoint_number=k)

            if failed:
                converged = False
            else:
                R_means = get_R_mean_dist(nf, self.eps, mu_colvec, self.M_test, N_test)
                converged = self.test_convergence(R_means.numpy(), alpha)
            last_ind = opt_it_df["iteration"] == k * num_iters

            opt_it_df.loc[last_ind, "converged"] = converged
            self._save_epi_opt(ckpt_dir, opt_it_df, cs, etas)
            opt_it_dfs = [opt_it_df]

            if k < K:
                if np.isnan(cost):
                    break
                # Check for convergence if early stopping.
                if stop_early and converged:
                    break

                # Update eta and c
                eta = eta + c * R
                norms_k = get_R_norm_dist(nf, self.eps, mu_colvec, self.M_norm, N)
                t, p = ttest_ind(
                    norms_k.numpy(), gamma * norms.numpy(), equal_var=False
                )
                u = np.random.rand(1)
                if u < 1 - p / 2.0 and t > 0.0:
                    c = beta * c
                norms = norms_k

        time_per_it = time2 - time1
        if save_movie_data:
            np.savez(
                os.path.join(ckpt_dir,"movie_data.npz"),
                zs=np.array(zs),
                log_q_zs=np.array(log_q_zs),
                time_per_it=time_per_it,
                iterations=np.arange(0, k * num_iters + 1, log_rate),
            )
        else:
            np.savez(os.path.join(ckpt_dir, "timing.npz"), time_per_it=time_per_it)

        # Save hyperparameters.
        self.aug_lag_hps = aug_lag_hps

        # Return optimized distribution.
        q_theta = Distribution(nf, self.parameters)
        q_theta.set_batch_norm_trainable(False)

        return q_theta, opt_it_dfs[0], ckpt_dir, failed

    def get_epi_df(self):
        base_path = os.path.join("data", "epi", self.name)
        next_listdir = [os.path.join(base_path, f) for f in os.listdir(base_path)]
        init_paths = [f for f in next_listdir if os.path.isdir(f)]
        dfs = []
        print('base_path', base_path)
        for init_path in init_paths:
            print('init_path', init_path)
            init = get_dir_index(os.path.join(init_path, "init.pkl"))
            if init is None: 
                continue
            next_listdir = [os.path.join(init_path, f) for f in os.listdir(init_path)]
            arch_paths = [f for f in next_listdir if os.path.isdir(f)]
            for arch_path in arch_paths:
                print('arch_path', arch_path)
                arch = get_dir_index(os.path.join(arch_path, "arch.pkl"))
                if arch is None: 
                    continue
                next_listdir = [os.path.join(arch_path, f) for f in os.listdir(arch_path)]
                ep_paths = [f for f in next_listdir if os.path.isdir(f)]
                for ep_path in ep_paths:
                    print('ep_path', ep_path)
                    ep = get_dir_index(os.path.join(ep_path, "ep.pkl"))
                    if ep is None: 
                        continue
                    next_listdir = [os.path.join(ep_path, f) for f in os.listdir(ep_path)]
                    AL_hp_paths = [f for f in next_listdir if os.path.isdir(f)]
                    for AL_hp_path in AL_hp_paths:
                        print('AL_hp_path', AL_hp_path)
                        print('list dir')
                        print(os.listdir(AL_hp_path))
                        AL_hps = get_dir_index(os.path.join(AL_hp_path, "AL_hps.pkl"))
                        print('AL_hps', AL_hps)
                        if AL_hps is None: 
                            print('******continuing!*****')
                            #continue
                        opt_data_file = os.path.join(AL_hp_path, "opt_data.csv")
                        if os.path.exists(opt_data_file):
                            df = pd.read_csv(opt_data_file)
                            df['path'] = AL_hp_path

                            df['init'] = df.shape[0]*[init]
                            df['arch'] = df.shape[0]*[arch]
                            df['EP'] = df.shape[0]*[ep]
                            df['AL_hps'] = df.shape[0]*[AL_hps]
                            dfs.append(df)
                        else:
                            print('******opt_data_file DNE*****')
                        print('dfs', dfs)
        return pd.concat(dfs)

    def epi_opt_movie(self, path):
        """Generate video of EPI optimization.

        :param path: Path to folder with optimization data.
        :type param: str
        """
        D = self.D
        palette = sns.color_palette()
        fontsize = 22

        z_filename = os.path.join(path, "movie_data.npz")
        opt_data_filename = os.path.join(path, "opt_data.csv")
        # Load zs for optimization.
        if os.path.exists(z_filename):
            z_file = np.load(z_filename)
        else:
            raise IOError("File %s does not exist." % z_filename)
        if os.path.exists(opt_data_filename):
            opt_data_df = pd.read_csv(opt_data_filename)
        else:
            raise IOError("File %s does not exist." % opt_data_filename)

        zs = z_file["zs"]
        log_q_zs = z_file["log_q_zs"]
        iters = z_file["iterations"]
        N_frames = len(iters)
        Hs = opt_data_df["H"]
        if (len(Hs) < N_frames) or (
            not np.isclose(iters, opt_data_df["iteration"][:N_frames]).all()
        ):
            raise IOError("opt_data.csv incompatible with movie_data.npz.")
        R_keys = []
        for key in opt_data_df.columns:
            if "R" in key:
                R_keys.append(key)
        m = len(R_keys)
        R = opt_data_df[R_keys].to_numpy()

        _iters = [iters[0]]
        _Hs = [Hs[0]]
        z = zs[0]
        log_q_z = log_q_zs[0]
        ylab_x = -0.075
        ylab_y = 0.6

        if not (self.name == "lds_2D"):
            iter_rows = 3
            # z_labels = [param.name for param in self.parameters]
            z_labels = ["z%d" % d for d in range(1, self.D + 1)]
            fig, axs = plt.subplots(D + iter_rows, D, figsize=(9, 12))
            H_ax = plt.subplot(D + iter_rows, 1, 1)
        else:
            z_labels = [r"$a_{11}$", r"$a_{12}$", r"$a_{21}$", r"$a_{22}$"]
            fig, axs = plt.subplots(4, 8, figsize=(16, 8))
            H_ax = plt.subplot(4, 2, 1)
            mode1s = []
            mode2s = []
            wsize = 100

        # Entropy lines
        x_end = 1.25 * iters[-1]
        opt_y_shiftx = -0.05
        num_iters = opt_data_df[opt_data_df["k"] == 1]["iteration"].max()
        K = opt_data_df["k"].max()
        log_rate = opt_data_df["iteration"][1]
        xticks = num_iters * np.arange(K + 1)
        H_ax.set_xlim(0, x_end)
        min_H, max_H = np.min(Hs), np.max(Hs)
        H_ax.set_ylim(min_H, max_H)
        (H_line,) = H_ax.plot(_iters, _Hs, c=palette[0])
        H_ax.set_ylabel(r"$H(q_\theta)$", rotation="horizontal", fontsize=fontsize)
        H_ax.yaxis.set_label_coords(ylab_x + opt_y_shiftx, ylab_y)
        H_ax.set_xticks(xticks)
        H_ax.set_xticklabels(len(xticks) * [""])
        H_ax.spines["bottom"].set_bounds(0, iters[-1])
        H_ax.spines["right"].set_visible(False)
        H_ax.spines["top"].set_visible(False)

        # Constraint lines
        if not (self.name == "lds_2D"):
            R_ax = plt.subplot(D + iter_rows, 1, 2)
        else:
            R_ax = plt.subplot(4, 2, 3)
        R_ax.set_xlim(0, iters[-1])
        min_R, max_R = np.min(R), np.max(R)
        R_ax.set_xlim(0, x_end)
        R_ax.set_ylim(min_R, max_R)
        R_ax.set_ylabel(r"$R(q_\theta)$", rotation="horizontal", fontsize=fontsize)
        R_ax.yaxis.set_label_coords(ylab_x + opt_y_shiftx, ylab_y)
        R_ax.set_xlabel("iterations", fontsize=(fontsize - 2))
        R_ax.set_xticks(xticks)
        xticklabels = ["0"] + ["%dk" % int(xtick / 1000) for xtick in xticks[1:]]
        R_ax.set_xticklabels(xticklabels, fontsize=(fontsize - 4))
        R_ax.spines["bottom"].set_bounds(0, iters[-1])
        R_ax.spines["right"].set_visible(False)
        R_ax.spines["top"].set_visible(False)

        if not (self.name == "lds_2D"):
            for j in range(D):
                axs[2, j].axis("off")
        else:
            # Plot the matrices
            def get_lds_2D_modes(z, log_q_z):
                M = log_q_z.shape[0]

                mode1 = np.logical_and(z[:, 1] > 0.0, z[:, 2] < 0)
                if sum(mode1) == 0:
                    mode1 = np.zeros((2, 2))
                else:
                    mode1_inds = np.arange(M)[mode1]
                    mode1_ind = mode1_inds[np.argmax(log_q_z[mode1])]
                    mode1 = np.reshape(z[mode1_ind], (2, 2))

                mode2 = np.logical_and(z[:, 1] < 0.0, z[:, 2] > 0)
                if sum(mode2) == 0:
                    mode2 = np.zeros((2, 2))
                else:
                    mode2_inds = np.arange(M)[mode2]
                    mode2_ind = mode2_inds[np.argmax(log_q_z[mode2])]
                    mode2 = np.reshape(z[mode2_ind], (2, 2))

                return mode1, mode2

            sqmat_xlims1 = [-0.2, 1.25]
            sqmat_xlims2 = [-0.05, 1.4]
            sqmat_ylims = [-0.05, 1.4]
            mode1, mode2 = get_lds_2D_modes(z, log_q_z)
            mode1s.append(mode1)
            mode2s.append(mode2)
            lw = 5
            gray = 0.4 * np.ones(3)
            bfrac = 0.05
            mat_ax = plt.subplot(2, 4, 5)
            texts = plot_square_mat(
                mat_ax,
                mode1,
                c=gray,
                lw=lw,
                fontsize=24,
                bfrac=bfrac,
                title="mode 1",
                xlims=sqmat_xlims1,
                ylims=sqmat_ylims,
                text_c=palette[1],
            )
            mat_ax = plt.subplot(2, 4, 6)
            texts += plot_square_mat(
                mat_ax,
                mode2,
                c=gray,
                lw=lw,
                fontsize=24,
                bfrac=bfrac,
                title="mode 2",
                xlims=sqmat_xlims2,
                ylims=sqmat_ylims,
                text_c=palette[3],
            )
            mode1_vec = np.reshape(mode1, (4,))
            mode2_vec = np.reshape(mode2, (4,))

        R_lines = []
        _Rs = []
        for i in range(m):
            _Rs.append([R[0, i]])
            (R_line,) = R_ax.plot(
                _iters, _Rs[i], label=R_keys[i], c="k"
            )  # palette[i + 1])
            R_lines.append(R_line)
        # R_ax.legend(loc=9)

        lines = [H_line] + R_lines

        # Get axis limits
        ax_mins = []
        ax_maxs = []
        lb, ub = self._get_bounds()
        for i in range(D):
            if np.isneginf(lb[i]):
                ax_mins.append(np.min(zs[:, :, i]))
            else:
                ax_mins.append(lb[i])

            if np.isposinf(ub[i]):
                ax_maxs.append(np.max(zs[:, :, i]))
            else:
                ax_maxs.append(ub[i])

        # Collect scatters
        cmap = plt.get_cmap("viridis")
        scats = []
        if not (self.name == "lds_2D"):
            scat_i = iter_rows
            scat_j = 0
        else:
            scat_i = 0
            scat_j = 4

        for i in range(D - 1):
            for j in range(i + 1, D):
                ax = axs[i + scat_i][j + scat_j]
                scats.append(ax.scatter(z[:, j], z[:, i], c=log_q_zs[0], cmap=cmap))
                ax.set_xlim(ax_mins[j], ax_maxs[j])
                ax.set_ylim(ax_mins[i], ax_maxs[i])
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)

        kdes = []
        conts = []
        kde_scale_fac = 0.1
        nlevels = 10
        num_grid = 20
        for i in range(1, D):
            ax_len_i = ax_maxs[i] - ax_mins[i]
            grid_ys = np.linspace(ax_mins[i], ax_maxs[i], num_grid)
            for j in range(i):
                ax_len_j = ax_maxs[j] - ax_mins[j]
                grid_xs = np.linspace(ax_mins[j], ax_maxs[j], num_grid)
                ax = axs[i + scat_i][j + scat_j]
                kde = KernelDensity(
                    kernel="gaussian",
                    bandwidth=kde_scale_fac * (ax_len_i + ax_len_j) / 2.0,
                )
                _z = z[:, [j, i]]
                kde.fit(_z, log_q_z)
                z_grid = np.meshgrid(grid_xs, grid_ys)
                z_grid_mat = np.stack(
                    [
                        np.reshape(z_grid[0], (num_grid ** 2)),
                        np.reshape(z_grid[1], (num_grid ** 2)),
                    ],
                    axis=1,
                )
                scores_ij = kde.score_samples(z_grid_mat)
                scores_ij = np.reshape(scores_ij, (num_grid, num_grid))
                levels = np.linspace(np.min(scores_ij), np.max(scores_ij), 20)
                cont = ax.contourf(z_grid[0], z_grid[1], scores_ij, levels=levels)
                conts.append(cont)
                ax.set_xlim(ax_mins[i], ax_maxs[i])
                ax.set_ylim(ax_mins[j], ax_maxs[j])
                kdes.append(kde)

        for i in range(D):
            axs[i + scat_i][scat_j].set_ylabel(
                z_labels[i], rotation="horizontal", fontsize=fontsize
            )
            axs[i + scat_i][scat_j].yaxis.set_label_coords(D * ylab_x, ylab_y)
            axs[-1][i + scat_j].set_xlabel(z_labels[i], fontsize=fontsize)
            axs[i + scat_i][i + scat_j].set_xlim(ax_mins[i], ax_maxs[i])
            axs[i + scat_i][i + scat_j].set_ylim(ax_mins[i], ax_maxs[i])
            axs[i + scat_i][i + scat_j].spines["right"].set_visible(False)
            axs[i + scat_i][i + scat_j].spines["top"].set_visible(False)

        # Plot modes
        if self.name == "lds_2D":
            for i in range(D - 1):
                for j in range(i + 1, D):
                    (line,) = axs[i + scat_i, j + scat_j].plot(
                        mode1_vec[j], mode1_vec[i], "o", c=palette[1]
                    )
                    lines.append(line)
                    (line,) = axs[i + scat_i, j + scat_j].plot(
                        mode2_vec[j], mode2_vec[i], "o", c=palette[3]
                    )
                    lines.append(line)

        # Tick labels
        for i in range(D):
            for j in range(1, D):
                axs[i + scat_i, j + scat_j].set_yticklabels([])
        for i in range(D - 1):
            for j in range(D):
                axs[i + scat_i, j + scat_j].set_xticklabels([])

        def update(frame):
            _iters.append(iters[frame])
            _Hs.append(Hs[frame])
            for i in range(m):
                _Rs[i].append(R[frame, i])
            z = zs[frame]
            log_q_z = log_q_zs[frame]
            cvals = log_q_z - np.min(log_q_z)
            cvals = cvals / np.max(cvals)

            # Update entropy plot
            lines[0].set_data(_iters, _Hs)
            for i in range(m):
                lines[i + 1].set_data(_iters, _Rs[i])

            # Update modes.
            if self.name == "lds_2D":
                mode1, mode2 = get_lds_2D_modes(z, log_q_z)
                mode1s.append(mode1)
                mode2s.append(mode2)

                mode1_avg = np.mean(np.array(mode1s)[-wsize:, :], axis=0)
                mode2_avg = np.mean(np.array(mode2s)[-wsize:, :], axis=0)
                mode1_vec = np.reshape(mode1_avg, (4,))
                mode2_vec = np.reshape(mode2_avg, (4,))
                ind = 0
                for i in range(2):
                    for j in range(2):
                        texts[ind].set_text("%.1f" % mode1_avg[i, j])
                        texts[ind + 4].set_text("%.1f" % mode2_avg[i, j])
                        ind += 1

                ind = 0
                for i in range(D - 1):
                    for j in range(i + 1, D):
                        lines[1 + m + ind].set_data(mode1_vec[j], mode1_vec[i])
                        ind += 1
                        lines[1 + m + ind].set_data(mode2_vec[j], mode2_vec[i])
                        ind += 1

            # Update scatters
            _ind = 0
            for i in range(D - 1):
                for j in range(i + 1, D):
                    scats[_ind].set_offsets(np.stack((z[:, j], z[:, i]), axis=1))
                    scats[_ind].set_color(cmap(cvals))
                    _ind += 1

            while conts:
                cont = conts.pop(0)
                for coll in cont.collections:
                    coll.remove()

            _ind = 0
            for i in range(1, D):
                grid_ys = np.linspace(ax_mins[i], ax_maxs[i], num_grid)
                for j in range(i):
                    grid_xs = np.linspace(ax_mins[j], ax_maxs[j], num_grid)
                    kde = kdes[_ind]
                    _z = z[:, [j, i]]
                    kde.fit(_z, log_q_z)
                    z_grid = np.meshgrid(grid_xs, grid_ys)
                    z_grid_mat = np.stack(
                        [
                            np.reshape(z_grid[0], (num_grid ** 2)),
                            np.reshape(z_grid[1], (num_grid ** 2)),
                        ],
                        axis=1,
                    )
                    scores_ij = kde.score_samples(z_grid_mat)
                    scores_ij = np.reshape(scores_ij, (num_grid, num_grid))
                    ax = axs[i + scat_i][j + scat_j]
                    levels = np.linspace(np.min(scores_ij), np.max(scores_ij), 20)
                    cont = ax.contourf(z_grid[0], z_grid[1], scores_ij, levels=levels)
                    conts.append(cont)

            return lines + scats

        if self.name == "lds_2D":
            frames = []
            skip = False
            logs_per_k = num_iters // log_rate
            for k in range(K):
                for iter in range(0, logs_per_k):
                    if k == 0:
                        frames += 4 * [k * logs_per_k + iter]
                    elif 1 <= k and k <= 2:
                        frames += 2 * [k * logs_per_k + iter]
                    else:
                        if not skip:
                            frames += [k * logs_per_k + iter]
                        skip = not skip
        else:
            frames = range(N_frames)

        ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)

        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=30, metadata=dict(artist="Me"), bitrate=1800)

        ani.save(os.path.join(path,"epi_opt.mp4"), writer=writer)
        return None

    def test_convergence(self, R_means, alpha, verbose=False):
        """Tests convergence of EPI constraints.

        :param R_means: Emergent property statistic means.
        :type R_means: np.ndarray
        :param alpha: P-value threshold.
        :type alpha: float
        """
        M, m = R_means.shape
        gt = np.sum(R_means > 0.0, axis=0).astype(np.float32)
        lt = np.sum(R_means < 0.0, axis=0).astype(np.float32)
        p_vals = 2 * np.minimum(gt / M, lt / M)
        if verbose:
            print(p_vals, alpha / m)
        return np.prod(p_vals > (alpha / m))

    def _opt_it_df(self, k, it, H, cost, R, log_rate, R_keys):
        d = {
            "k": k, 
            "iteration": it, 
            "H": H, 
            "cost": cost, 
            "converged": None
        }
        d.update(zip(R_keys, list(R)))
        return pd.DataFrame(d, index=[it//log_rate])

    def _save_epi_opt(self, save_path, opt_df, etas, cs):
        np.savez(os.path.join(save_path, "opt_data.npz"), etas=etas, cs=cs)
        opt_df.to_csv(os.path.join(save_path, "opt_data.csv"))

    def get_epi_path(self, init_params, nf, mu, AL_hps, eps_name=None):
        if eps_name is not None:
            _eps_name = eps_name
        else:
            if self.eps is not None:
                _eps_name = self.eps.__name__
            else:
                raise AttributeError("Model.eps is not set.")
        init_hash = get_hash([init_params["mu"], init_params["Sigma"], nf.lb, nf.ub])
        ep_hash = get_hash([_eps_name, mu])

        base_path = os.path.join("data", "epi")

        epi_path = os.path.join(
            base_path,
            self.name,
            init_hash,
            nf.to_string(),
            ep_hash,
            AL_hps.to_string(),
        )

        if not os.path.exists(epi_path):
            os.makedirs(epi_path)

        init_index = {
            "mu": init_params["mu"],
            "Sigma": init_params["Sigma"],
            "lb": nf.lb,
            "ub": nf.ub,
        }
        init_index_file = os.path.join(
            base_path,
            self.name, 
            init_hash,
            "init.pkl"
        )

        arch_index = {
            "arch_type": nf.arch_type,
            "D": nf.D,
            "num_stages": nf.num_stages,
            "num_layers": nf.num_layers,
            "num_units": nf.num_units,
            "batch_norm": nf.batch_norm,
            "bn_momentum": nf.bn_momentum,
            "post_affine": nf.post_affine,
            "lb": nf.lb,
            "ub": nf.ub,
            "random_seed": nf.random_seed,
        }
        arch_index_file = os.path.join(
            base_path,
            self.name,
            init_hash,
            nf.to_string(),
            "arch.pkl",
        )

        ep_index = {
            "name": _eps_name,
            "mu": mu,
        }
        ep_index_file = os.path.join(
            base_path,
            self.name,
            init_hash,
            nf.to_string(),
            ep_hash,
            "ep.pkl",
        )

        AL_hp_index = {
            "N": AL_hps.N,
            "lr": AL_hps.lr,
            "c0": AL_hps.c0,
            "gamma": AL_hps.gamma,
            "beta": AL_hps.beta,
        }
        AL_hp_index_file = os.path.join(
            base_path,
            self.name,
            init_hash,
            nf.to_string(),
            ep_hash,
            AL_hps.to_string(),
            "Al_hps.pkl",
        )

        indexes = [init_index, arch_index, ep_index, AL_hp_index]
        index_files = [
            init_index_file,
            arch_index_file,
            ep_index_file,
            AL_hp_index_file,
        ]

        for index, index_file in zip(indexes, index_files):
            exists = set_dir_index(index, index_file)

        # if final index is set, this EPI opt has been run before.
        return epi_path, exists

    def get_epi_dist(self, df_row):
        init = df_row["init"]
        ep = df_row["EP"]

        k = int(df_row["k"])
        init_params = {"mu":init["mu"], "Sigma":init["Sigma"]}
        nf = self._df_row_to_nf(df_row)
        mu = ep["mu"]
        aug_lag_hps = self._df_row_to_al_hps(df_row)
        return self._get_epi_dist(k, init_params, nf, mu, aug_lag_hps)

    def _df_row_to_nf(self, df_row):
        arch = df_row['arch']
        nf = NormalizingFlow(
            arch_type=arch["arch_type"],
            D=arch["D"],
            num_stages=arch["num_stages"],
            num_layers=arch["num_layers"],
            num_units=arch["num_units"],
            batch_norm=arch["batch_norm"],
            bn_momentum=arch["bn_momentum"],
            post_affine=arch["post_affine"],
            bounds=(arch["lb"], arch["ub"]),
            random_seed=arch["random_seed"],
        )
        return nf

    def _df_row_to_al_hps(self, df_row):
        AL_hps = df_row['AL_hps']
        aug_lag_hps = AugLagHPs(
            N=AL_hps["N"],
            lr=AL_hps["lr"],
            c0=AL_hps["c0"],
            gamma=AL_hps["gamma"],
            beta=AL_hps["beta"],
        )
        return aug_lag_hps


    def _get_epi_dist(self, k, init_params, nf, mu, aug_lag_hps):
        if k is not None:
            if type(k) is not int:
                raise TypeError(format_type_err_msg("Model.load_epi_dist", "k", k, int))

        optimizer = tf.keras.optimizers.Adam(aug_lag_hps.lr)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=nf)
        ckpt_dir, exists = self.get_epi_path(init_params, nf, mu, aug_lag_hps)
        if not exists:
            return None
        ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt_state is not None:
            ckpts = ckpt_state.all_model_checkpoint_paths
        else:
            raise ValueError("No checkpoints found.")

        num_ckpts = len(ckpts)
        if k >= num_ckpts and k < -num_ckpts:
            raise ValueError("Index of checkpoint 'k' out of range.")

        if k >= 0:
            status = checkpoint.restore(ckpts[k])
        else:
            status = checkpoint.restore(ckpts[num_ckpts+k])
        status.expect_partial()
        q_theta = Distribution(nf, self.parameters)
        return q_theta

    def get_convergence_epoch(
        self, init_params, nf, mu, aug_lag_hps, alpha=0.05, nu=0.1, mu_test=None
    ):

        if mu_test is not None:
            _mu = np_column_vec(mu_test).astype(np.float32).T
        else:
            _mu = np_column_vec(mu).astype(np.float32).T
        N_test = int(nu * aug_lag_hps.N)

        optimizer = tf.keras.optimizers.Adam(aug_lag_hps.lr)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=nf)
        ckpt_dir, exists = self.get_epi_path(init_params, nf, mu, aug_lag_hps)
        if not exists:
            return None, None, None

        ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt_state is not None:
            ckpts = ckpt_state.all_model_checkpoint_paths
        else:
            raise ValueError("No checkpoints found.")
        num_ckpts = len(ckpts)

        best_k = None
        best_H = None
        converged = False
        for k in range(num_ckpts):
            status = checkpoint.restore(ckpts[k])
            status.expect_partial()

            m = _mu.shape[1]
            z, log_q_z = nf(self.M_test * N_test)
            T_x = self.eps(z)
            T_x = tf.reshape(T_x, (self.M_test, N_test, m))
            R_means = tf.reduce_mean(T_x, axis=1) - _mu

            # R_means = get_R_mean_dist(nf, self.eps, _mu, self.M_test, N_test)
            _converged = self.test_convergence(R_means.numpy(), alpha, verbose=True)
            if _converged:
                H = -np.mean(log_q_z.numpy())
                if best_H is None or best_H < H:
                    best_k = k
                    best_H = H
                converged = True
        return best_k, converged, best_H

    def parameter_check(self, parameters, verbose=False):
        """Check that model parameter list has no duplicates and valid bounds.

        :param parameters: List of :obj:`epi.models.Parameter`.
        :type parameters: list
        :param verbose: Print rationale for check failure if True, defaults to False.
        :type verbose: bool, optional
        :return: True if parameter list is valid.
        :rtype: bool
        """
        d = dict()
        for param in parameters:
            name = param.name
            if name in d:
                if verbose:
                    print("Warning: Duplicate parameter %s in Model.parameters." % name)
                return False
            else:
                d[name] = True

            lb = param.lb
            ub = param.ub
            for i in range(param.D):
                lb_i = lb[i]
                ub_i = ub[i]
                if lb_i == ub_i:
                    if verbose:
                        print(
                            "Warning: Left bound equal to right bound for parameter %s."
                            % name
                        )
                    return False
                elif lb_i > ub_i:
                    if verbose:
                        print(
                            "Warning: Left bound greater than right bound for parameter %s."
                            % name
                        )
                    return False

        return True


class Distribution(object):
    """Distribution class with numpy UI, and tensorflow-enabled methods.

    Obtain samples, log densities, gradients and Hessians of a distribution
    defined by a normalizing flow optimized via tensorflow.

    :param nf: Normalizing flow trained via tensorflow.
    :type nf: :obj:`epi.normalizing_flows.NormalizingFlow`
    :param parameters: List of :obj:`epi.models.Parameter`. Defaults to z1, ..
    :type parameters: list, optional
    """

    def __init__(self, nf, parameters=None):
        self._set_nf(nf)
        self.D = self.nf.D
        self._set_parameters(parameters)

    def __call__(self, N):
        z, _ = self.nf(N)
        return z.numpy()

    def _set_nf(self, nf):
        self.nf = nf

    def _set_parameters(self, parameters):
        if parameters is None:
            parameters = [Parameter("z%d" % (i + 1), 1) for i in range(self.D)]
            self.parameters = parameters
        elif type(parameters) is not list:
            raise TypeError(format_type_err_msg(self, "parameters", parameters, list))
        else:
            for parameter in parameters:
                if not parameter.__class__.__name__ == "Parameter":
                    raise TypeError(
                        format_type_err_msg(self, "parameter", parameter, Parameter)
                    )
            self.parameters = parameters

    def sample(self, N):
        """Sample N times.

        :param N: Number of samples.
        :type N: int:
        :returns: N samples.
        :rtype: np.ndarray
        """
        if type(N) is not int:
            raise TypeError(self, "N", N, int)
        elif N < 1:
            raise ValueError(
                "Distribution.sample must be called with positive int not %d." % N
            )
        return self.__call__(N)

    def log_prob(self, z):
        """Calculates log probability of samples from distribution.

        :param z: Samples from distribution.
        :type z: np.ndarray
        :returns: Log probability of samples.
        :rtype: np.ndarray
        """
        z = self._set_z_type(z)
        return self.nf.trans_dist.log_prob(z).numpy()

    def gradient(self, z):
        """Calculates the gradient :math:`\\nabla_z \\log p(z))`.

        :param z: Samples from distribution.
        :type z: np.ndarray
        :returns: Gradient of log probability with respect to z.
        :rtype: np.ndarray
        """
        z = self._set_z_type(z)
        z = tf.Variable(initial_value=z, trainable=True)
        grad_z = self._gradient(z)
        del z  # Get rid of dummy variable.
        return grad_z.numpy()

    @tf.function
    def _gradient(self, z):
        with tf.GradientTape() as tape:
            log_q_z = self.nf.trans_dist.log_prob(z)
        return tape.gradient(log_q_z, z)

    def hessian(self, z):
        """Calculates the Hessian :math:`\\frac{\\partial^2 \\log p(z)}{\\partial z \\partial z^\\top}`.

        :param z: Samples from distribution.
        :type z: np.ndarray
        :returns: Hessian of log probability with respect to z.
        :rtype: np.ndarray
        """
        z = self._set_z_type(z)
        z = tf.Variable(initial_value=z, trainable=True)
        hess_z = self._hessian(z)
        del z  # Get rid of dummy variable.
        return hess_z.numpy()

    @tf.function
    def _hessian(self, z):
        with tf.GradientTape(persistent=True) as tape:
            log_q_z = self.nf.trans_dist.log_prob(z)
            dldz = tape.gradient(log_q_z, z)
        return tape.batch_jacobian(dldz, z)

    def _set_z_type(self, z):
        if type(z) is list:
            z = np.ndarray(z)
        if type(z) is not np.ndarray:
            raise TypeError(self, z, "z", np.ndarray)
        z = z.astype(np.float32)
        return z

    def plot_dist(self, N=200, kde=True):
        z = self.sample(N)
        log_q_z = self.log_prob(z)
        df = pd.DataFrame(z)
        # iterate over parameters to create label_names
        z_labels = []
        for param in self.parameters:
            if param.D == 1:
                z_labels.append(param.name)
            else:
                z_labels.extend([str(param.name) + str(i) for i in range(param.D)])

        df.columns = z_labels
        df["log_q_z"] = log_q_z

        log_q_z_std = log_q_z - np.min(log_q_z)
        log_q_z_std = log_q_z_std / np.max(log_q_z_std)
        cmap = plt.get_cmap("viridis")
        g = sns.PairGrid(df, vars=z_labels)
        g = g.map_upper(plt.scatter, color=cmap(log_q_z_std))
        if kde:
            g = g.map_diag(sns.kdeplot)
            g = g.map_lower(sns.kdeplot)
        return g

    def set_batch_norm_trainable(self, trainable):
        bijectors = self.nf.trans_dist.bijector.bijectors
        for bijector in bijectors:
            if type(bijector).__name__ == "BatchNormalization":
                bijector._training = trainable
        return None


@tf.function
def two_dim_T_x_batch(nf, eps, M, N, m):
    z, _ = nf(M * N)
    T_x = eps(z)
    T_x = tf.reshape(T_x, (M, N, m))
    return T_x


@tf.function
def get_R_norm_dist(nf, eps, mu, M, N):
    m = mu.shape[1]
    T_x = two_dim_T_x_batch(nf, eps, M, N, m)
    return tf.reduce_sum(tf.square(tf.reduce_mean(T_x, axis=1) - mu), axis=1)


@tf.function
def get_R_mean_dist(nf, eps, mu, M, N):
    m = mu.shape[1]
    T_x = two_dim_T_x_batch(nf, eps, M, N, m)
    return tf.reduce_mean(T_x, axis=1) - mu


def format_opt_msg(k, i, cost, H, R):
    s1 = "" if cost < 0.0 else " "
    s2 = "" if H < 0.0 else " "
    args = (k, i, s1, cost, s2, H, np.sum(np.square(R)))
    return "EPI(k=%2d,i=%4d): cost %s%.2E, H %s%.2E, |R|^2 %.2E" % args
