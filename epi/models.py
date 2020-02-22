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
    :param bounds: Lower and upper bound of variable, defaults to (np.NINF, np.PINF).
    :type bounds: (np.float, np.float), optional
    """

    def __init__(self, name, bounds=(np.NINF, np.PINF)):
        """Constructor method."""
        self._set_name(name)
        self._set_bounds(bounds)

    def _set_name(self, name):
        if type(name) is not str:
            raise TypeError(format_type_err_msg(self, "name", name, str))
        self.name = name

    def _set_bounds(self, bounds):
        _type = type(bounds)
        if _type in [list, tuple]:
            len_bounds = len(bounds)
            if _type is list:
                bounds = tuple(bounds)
        elif _type is np.ndarray:
            len_bounds = bounds.shape[0]
            bounds = (bounds[0], bounds[1])
        else:
            raise TypeError(
                "Parameter argument bounds must be tuple, list, or numpy array not %s."
                % _type.__name__
            )

        if len_bounds != 2:
            raise ValueError("Parameter bounds arg must be length 2.")

        lb = bounds[0]
        ub = bounds[1]
        if not isinstance(lb, REAL_NUMERIC_TYPES):
            raise TypeError("Lower bound has type %s, not numeric." % type(lb))
        if not isinstance(ub, REAL_NUMERIC_TYPES):
            raise TypeError("Upper bound has type %s, not numeric." % type(ub))

        if lb > ub:
            raise ValueError(
                "Parameter %s lower bound is greater than upper bound." % self.name
            )
        elif lb == ub:
            raise ValueError(
                "Parameter %s lower bound is equal to upper bound." % self.name
            )

        self.bounds = bounds


class Model(object):
    """Model to run emergent property inference on.  To run EPI on a model:

    #. Initialize an :obj:epi.models.Model with a list of :obj:`epi.models.Parameter`.
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
        self.D = len(parameters)

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
        for arg in args:
            for param in self.parameters:
                if param.name == arg:
                    _parameters.append(param)
                    self.parameters.remove(param)
        self.parameters = _parameters

        def _eps(z):
            zs = tf.unstack(z[:, :], axis=1)
            return eps(*zs)

        self.eps = _eps
        self.eps.__name__ = eps.__name__

        z = tf.keras.Input(shape=(self.D))
        T_z = self.eps(z)
        T_z_shape = T_z.shape
        if len(T_z_shape) != 2:
            raise ValueError("Method eps must return tf.Tensor of dimension (N, D).")
        self.m = T_z_shape[1]
        return None

    def _get_bounds(self,):
        D = len(self.parameters)
        lb = np.zeros((D,))
        ub = np.zeros((D,))
        for i, param in enumerate(self.parameters):
            lb[i] = param.bounds[0]
            ub[i] = param.bounds[1]
        return (lb, ub)

    def epi(
        self,
        mu,
        arch_type="autoregressive",
        num_stages=1,
        num_layers=2,
        num_units=None,
        batch_norm=True,
        bn_momentum=0.99,
        post_affine=True,
        random_seed=1,
        init_type="iso_gauss",
        init_params={"loc": 0.0, "scale": 1.0},
        K=10,
        num_iters=2000,
        N=500,
        lr=1e-3,
        c0=1.0,
        gamma=0.25,
        beta=4.0,
        alpha=0.05,
        nu=0.1,
        stop_early=False,
        log_rate=100,
        verbose=False,
        save_movie_data=False,
    ):
        """Runs emergent property inference for this model with mean parameter :math:`\\mu`.


        :param mu: Mean parameter of the emergent property.
        :type mu: np.ndarray
        :param arch_type: :math:`\\in` :obj:`['autoregressive', 'coupling']`, defaults to :obj:`'autoregressive'`.
        :type arch_type: str, optional
        :param num_stages: Number of coupling or autoregressive stages.
        :type num_stages: int, optional
        :param num_layers: Number of neural network layer per conditional.
        :type num_layers: int, optional
        :param num_units: Number of units per layer.
        :type num_units: int, optional
        :param batch_norm: Use batch normalization between stages, defaults to True.
        :type batch_norm: bool, optional
        :param bn_momentum: Batch normalization momentum parameter, defaults to 0.99.
        :type bn_momentrum: float, optional
        :param post_affine: Shift and scale following main transform, defaults to True.
        :type post_affine: bool, optional
        :param random_seed: Random seed of architecture parameters, defaults to 1.
        :type random_seed: int, optional
        :param init_type: :math:`\\in` :obj:`['iso_gauss', 'gaussian']`.
        :type init_type: str, optional
        :param init_params: Parameters according to :obj:`init_type`.
        :type init_params: dict, optional
        :param K: Number of augmented Lagrangian iterations, defaults to 10.
        :type K: int, float, optional
        :param num_iters: Number of optimization iterations, defaults to 2000.
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
        nf.initialize(init_type, init_params)

        # Checkpoint the initialization.
        optimizer = tf.keras.optimizers.Adam(lr)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=nf)
        ckpt_dir = self.get_save_path(mu, nf, aug_lag_hps)
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

        M_test = 200
        N_test = int(nu * N)
        M_norm = 200
        # Initialize augmented Lagrangian parameters eta and c.
        eta, c = np.zeros((self.m,), np.float32), c0
        etas, cs = np.zeros((K, self.m)), np.zeros((K,))

        # Initialize optimization data frame.
        z, log_q_z = nf(N)
        H_0, R_0, _, _ = aug_lag_vars(z, log_q_z, self.eps, mu, N)
        cost_0 = -H_0 + np.dot(eta, R_0) + np.sum(np.square(R_0))
        R_keys = ["R%d" % (i + 1) for i in range(self.m)]
        opt_it_dfs = [self._opt_it_df(0, 0, H_0.numpy(), R_0.numpy(), R_keys)]

        # Record samples for movie.
        if save_movie_data:
            N_save = 200
            zs = [z.numpy()[:N_save, :]]
            log_q_zs = [log_q_z.numpy()[:N_save]]

        # Measure initial R norm distribution.
        mu_colvec = np_column_vec(mu).astype(np.float32).T
        norms = get_R_norm_dist(nf, self.eps, mu_colvec, M_norm, N)

        # EPI optimization
        print(format_opt_msg(0, 0, cost_0, H_0, R_0), flush=True)
        for k in range(1, K + 1):
            etas[k - 1], cs[k - 1], eta, c
            for i in range(1, num_iters + 1):
                time1 = time.time()
                cost, H, R, z, log_q_z = train_step(eta, c)
                time2 = time.time()
                if i % log_rate == 0:
                    if verbose:
                        print(format_opt_msg(k, i, cost, H, R), flush=True)
                    iter = (k - 1) * num_iters + i
                    opt_it_dfs.append(
                        self._opt_it_df(k, iter, H.numpy(), R.numpy(), R_keys)
                    )
                    if save_movie_data:
                        zs.append(z.numpy()[:N_save, :])
                        log_q_zs.append(log_q_z.numpy()[:N_save])
                if (np.isnan(cost)):
                    break
            if not verbose:
                print(format_opt_msg(k, i, cost, H, R), flush=True)

            # Save epi optimization data following aug lag iteration k.
            opt_it_df = pd.concat(opt_it_dfs, ignore_index=True)
            manager.save(checkpoint_number=k)

            R_means = get_R_mean_dist(nf, self.eps, mu_colvec, M_test, N_test)
            converged = self.test_convergence(R_means.numpy(), alpha)
            last_ind = opt_it_df['iteration']==k*num_iters

            opt_it_df.loc[last_ind, 'converged'] = converged
            self._save_epi_opt(ckpt_dir, opt_it_df, cs, etas)
            opt_it_dfs = [opt_it_df]

            if k < K:
                if (np.isnan(cost)):
                    break
                # Check for convergence if early stopping.
                if stop_early and converged:
                    break

                # Update eta and c
                eta = eta + c * R
                norms_k = get_R_norm_dist(nf, self.eps, mu_colvec, M_norm, N)
                t, p = ttest_ind(
                    norms_k.numpy(), gamma * norms.numpy(), equal_var=False
                )
                u = np.random.rand(1)
                if u < 1 - p / 2.0 and t > 0.0:
                    c = beta * c
                norms = norms_k

        if save_movie_data:
            np.savez(
                ckpt_dir + "movie_data.npz",
                zs=np.array(zs),
                log_q_zs=np.array(log_q_zs),
                iterations=np.arange(0, k * num_iters + 1, log_rate),
            )

        # Save hyperparameters.
        self._save_hps(ckpt_dir, nf, aug_lag_hps, init_type, init_params)

        # Return optimized distribution.
        q_theta = Distribution(nf, self.parameters)

        return q_theta, opt_it_dfs[0], ckpt_dir

    def plot_epi_hpsearch(self, mu, alpha=0.05, nu=0.1):
        epi_dir = self.get_epi_path(mu)
        if (not os.path.exists(epi_dir)):
            raise IOError("Directory %s does not exist." % epi_dir)
        opt_dirs = os.listdir(epi_dir)
        n = len(opt_dirs)
        if (n == 0):
            raise IOError("No optimizations in %s." % epi_dir)
        Hs = []
        hp_dfs = []
        opt_dfs = []
        for i, opt_dir in enumerate(opt_dirs):
            print(i, opt_dir)
            # Parse hp file.
            hp_filename = epi_dir + opt_dir + '/hps.p'
            if (not os.path.exists(hp_filename)):
                print('skipping', hp_filename)
                continue
            hps = pickle.load(open(hp_filename, "rb"))
            hps['N'] = hps['aug_lag_hps'].N
            hps['lr'] = hps['aug_lag_hps'].lr
            hps['c0'] = hps['aug_lag_hps'].c0
            hps['gamma'] = hps['aug_lag_hps'].gamma
            del hps['aug_lag_hps']
            del hps['init_type']
            del hps['init_params']
            hp_dfs.append(pd.DataFrame(hps, index=[i]))

            # Parse opt data file.
            opt_filename = epi_dir + opt_dir + '/opt_data.csv'
            opt_df_i = pd.read_csv(opt_filename)
            opt_df_i['hp'] = opt_dir
            opt_dfs.append(opt_df_i)

            # Calculate evaluation statistics
            conv_its = opt_df_i['converged'] == True
            if (np.sum(conv_its) == 0):
                Hs.append(np.nan)
            else:
                Hs.append(np.max(opt_df_i.loc[conv_its, 'H']))

        hp_df = pd.concat(hp_dfs, sort=False)
        hp_df['H'] = np.array(Hs)
        opt_df = pd.concat(opt_dfs, sort=False)

        hps = opt_df['hp'].unique()
        
        # Plot optimization diagnostics.
        m = mu.shape[0]
        fig, axs = plt.subplots(m+1, 1, figsize=(10,m*3))
        for hp in hps:
            opt_df_hp = opt_df[opt_df['hp'] == hp]
            axs[0].plot(opt_df_hp['iteration'], opt_df_hp['H'], label=hp)
            axs[0].set_ylabel(r'$H(q_\theta)$')
            for i in range(m):
                axs[i+1].plot(opt_df_hp['iteration'], opt_df_hp['R%d' % (i+1)])
                axs[i+1].set_ylabel(r'$R(q_\theta)_{%d}$' % (i+1))
        plt.show()

        # Plot scatters of hyperparameters with stats.
        arch_types = hp_df['arch_type'].unique()
        dtypes = hp_df.dtypes
        Hisnan = hp_df['H'].isna()
        minH = hp_df['H'].min()
        numnan = Hisnan.sum()
        for arch_type in arch_types:
            nrows = 3
            ncols = 4
            fig, axs = plt.subplots(3, 4, figsize=(14, 14))
            hp_df_arch = hp_df[hp_df['arch_type'] == arch_type]
            ind = 1
            for i in range(nrows):
                axs[i][0].set_ylabel('H')
                for j in range(ncols):
                    col = hp_df.columns[ind]
                    ind += 1
                    #if (col in ['batch_norm', 'post_affine']):
                    #    continue
                    axs[i][j].scatter(hp_df_arch[col], hp_df_arch['H'])
                    axs[i][j].scatter(hp_df_arch[col][Hisnan].to_numpy(),
                                      (minH-1)*np.ones(numnan),
                                      c='r', marker='x')
                    axs[i][j].set_xlabel(col)
        plt.show()

        return hp_df, opt_df

    def _save_hps(self, ckpt_dir, nf, aug_lag_hps, init_type, init_params):
        """Save hyperparameters to save directory.

        :param ckpt_dir: Path the save directory.
        :type ckpt_dir: str
        :param nf: Normalizing flow.
        :type nf: :obj:`epi.normalizing_flows.NormalizingFlow`
        :param aug_lag_hps: Augmented Lagrangian hyperparameters.
        :type aug_lag_hps: :obj:`epi.util.AugLagHPs`
        """

        hps = {'arch_type':nf.arch_type,
               'num_stages':nf.num_stages,
               'num_layers':nf.num_layers,
               'num_units':nf.num_units,
               'batch_norm':nf.batch_norm,
               'bn_momentum':nf.bn_momentum,
               'post_affine':nf.post_affine,
               'random_seed':nf.random_seed,
               'init_type':init_type,
               'init_params':init_params,
               'aug_lag_hps':aug_lag_hps
               }
        pickle.dump(hps, open(ckpt_dir + 'hps.p', "wb"))


    def epi_opt_movie(self, path):
        """Generate video of EPI optimization.

        :param path: Path to folder with optimization data.
        :type param: str
        """
        D = len(self.parameters)
        palette = sns.color_palette()
        fontsize = 22

        z_filename = path + "movie_data.npz"
        opt_data_filename = path + "opt_data.csv"
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
        if (len(Hs) < N_frames) or (not np.isclose(iters, opt_data_df["iteration"][:N_frames]).all()):
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

        if (not (self.name == "lds_2D")):
            iter_rows = 3
            z_labels = [param.name for param in self.parameters]
            fig, axs = plt.subplots(D + iter_rows, D, figsize=(9, 12))
            H_ax = plt.subplot(D + iter_rows, 1, 1)
        else:
            z_labels = [r'$a_{11}$', r'$a_{12}$', r'$a_{21}$', r'$a_{22}$']
            fig, axs = plt.subplots(4, 8, figsize=(16, 8))
            H_ax = plt.subplot(4, 2, 1)
            mode1s = []
            mode2s = []
            wsize = 100

        # Entropy lines
        x_end = 1.25*iters[-1]
        num_iters = opt_data_df[opt_data_df['k']==1]['iteration'].max()
        K = opt_data_df['k'].max()
        log_rate = opt_data_df['iteration'][1]
        xticks = num_iters*np.arange(K+1)
        H_ax.set_xlim(0, x_end)
        min_H, max_H = np.min(Hs), np.max(Hs)
        H_ax.set_ylim(min_H, max_H)
        H_line, = H_ax.plot(_iters, _Hs, c=palette[0])
        H_ax.set_ylabel(r"$H(q_\theta)$", rotation="horizontal", fontsize=fontsize)
        H_ax.yaxis.set_label_coords(ylab_x, ylab_y)
        H_ax.set_xticks(xticks)
        H_ax.set_xticklabels(xticks)
        H_ax.spines['bottom'].set_bounds(0, iters[-1])
        H_ax.spines['right'].set_visible(False)
        H_ax.spines['top'].set_visible(False)

        # Constraint lines
        if (not (self.name == "lds_2D")):
            R_ax = plt.subplot(D + iter_rows, 1, 2)
        else:
            R_ax = plt.subplot(4, 2, 3)
        R_ax.set_xlim(0, iters[-1])
        min_R, max_R = np.min(R), np.max(R)
        R_ax.set_xlim(0, x_end)
        R_ax.set_ylim(min_R, max_R)
        R_ax.set_ylabel(r"$R(q_\theta)$", rotation="horizontal", fontsize=fontsize)
        R_ax.yaxis.set_label_coords(ylab_x, ylab_y)
        R_ax.set_xlabel("iterations", fontsize=(fontsize-2))
        R_ax.set_xticks(xticks)
        R_ax.set_xticklabels(xticks)
        R_ax.spines['bottom'].set_bounds(0, iters[-1])
        R_ax.spines['right'].set_visible(False)
        R_ax.spines['top'].set_visible(False)

        if (not (self.name == "lds_2D")):
            for j in range(D):
                axs[2, j].axis("off")
        else:
            # Plot the matrices
            def get_lds_2D_modes(z, log_q_z):
                M = log_q_z.shape[0]
                mode1 = np.logical_and(z[:,1] > 0., z[:,2] < 0)
                mode1_inds = np.arange(M)[mode1]
                mode2 = np.logical_and(z[:,1] < 0., z[:,2] > 0)
                mode2_inds = np.arange(M)[mode2]
                mode1_ind = mode1_inds[np.argmax(log_q_z[mode1])]
                mode2_ind = mode2_inds[np.argmax(log_q_z[mode2])]
                return np.reshape(z[mode1_ind], (2,2)), np.reshape(z[mode2_ind], (2,2))

            sqmat_xlims1 = [-.2, 1.25]
            sqmat_xlims2 = [-.05, 1.4]
            sqmat_ylims = [-0.05, 1.4]
            mode1, mode2 = get_lds_2D_modes(z, log_q_z)
            mode1s.append(mode1)
            mode2s.append(mode2)
            lw=5
            gray = 0.4*np.ones(3)
            bfrac=0.05
            mat_ax = plt.subplot(2, 4, 5)
            texts = plot_square_mat(mat_ax, mode1, c=gray, lw=lw, fontsize=24, bfrac=bfrac, title="mode 1",
                                    xlims=sqmat_xlims1, ylims=sqmat_ylims)
            mat_ax = plt.subplot(2, 4, 6)
            texts += plot_square_mat(mat_ax, mode2, c=gray, lw=lw, fontsize=24, bfrac=bfrac, title="mode 2",
                                     xlims=sqmat_xlims2, ylims=sqmat_ylims)

        R_lines = []
        _Rs = []
        for i in range(m):
            _Rs.append([R[0, i]])
            R_line, = R_ax.plot(_iters, _Rs[i], label=R_keys[i], c=palette[i + 1])
            R_lines.append(R_line)
        R_ax.legend(loc=9)

        lines = [H_line] + R_lines

        # Get axis limits
        ax_mins = []
        ax_maxs = []
        for i in range(D):
            bounds = self.parameters[i].bounds
            if np.isneginf(bounds[0]):
                ax_mins.append(np.min(zs[:, :, i]))
            else:
                ax_mins.append(bounds[0])

            if np.isposinf(bounds[1]):
                ax_maxs.append(np.max(zs[:, :, i]))
            else:
                ax_maxs.append(bounds[1])

        # Collect scatters
        cmap = plt.get_cmap("viridis")
        scats = []
        if (not (self.name == "lds_2D")):
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
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

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
                z_grid_mat = np.stack([np.reshape(z_grid[0], (num_grid**2)),
                                   np.reshape(z_grid[1], (num_grid**2))],
                                  axis=1)
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
                z_labels[i], rotation="horizontal", fontsize=fontsize,
            )
            axs[i + scat_i][scat_j].yaxis.set_label_coords(D * ylab_x, ylab_y)
            axs[-1][i+scat_j].set_xlabel(z_labels[i], fontsize=fontsize)
            axs[i+scat_i][i+scat_j].set_xlim(ax_mins[i], ax_maxs[i])
            axs[i+scat_i][i+scat_j].set_ylim(ax_mins[i], ax_maxs[i])
            axs[i+scat_i][i+scat_j].spines['right'].set_visible(False)
            axs[i+scat_i][i+scat_j].spines['top'].set_visible(False)

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

            # Update scatters
            _ind = 0
            for i in range(D - 1):
                for j in range(i + 1, D):
                    scats[_ind].set_offsets(np.stack((z[:, j], z[:, i]), axis=1))
                    scats[_ind].set_color(cmap(cvals))
                    _ind += 1

            # Update modes.
            if (self.name == "lds_2D"):
                mode1, mode2 = get_lds_2D_modes(z, log_q_z)
                mode1s.append(mode1)
                mode2s.append(mode2)
                mode1_avg = np.mean(np.array(mode1s)[-wsize:,:], axis=0)
                mode2_avg = np.mean(np.array(mode2s)[-wsize:,:], axis=0)
                ind = 0
                for i in range(2):
                    for j in range(2):
                        texts[ind].set_text('%.1f' % mode1_avg[i,j])
                        texts[ind+4].set_text('%.1f' % mode2_avg[i,j])
                        ind += 1

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
                    z_grid_mat = np.stack([np.reshape(z_grid[0], (num_grid**2)),
                                       np.reshape(z_grid[1], (num_grid**2))],
                                      axis=1)
                    scores_ij = kde.score_samples(z_grid_mat)
                    scores_ij = np.reshape(scores_ij, (num_grid, num_grid))
                    ax = axs[i + scat_i][j + scat_j]
                    levels = np.linspace(np.min(scores_ij), np.max(scores_ij), 20)
                    cont = ax.contourf(z_grid[0], z_grid[1], scores_ij, levels=levels)
                    conts.append(cont)

            return lines + scats

        if (self.name == "lds_2D"):
            frames = []
            skip = False
            logs_per_k = num_iters // log_rate
            for k in range(K):
                for iter in range(0, logs_per_k):
                    if (k==0):
                        frames += 4*[k*logs_per_k + iter]
                    elif (1 <= k and k <= 2):
                        frames += 2*[k*logs_per_k + iter]
                    else:
                        if not skip:
                            frames += [k*logs_per_k + iter]
                        skip = not skip
        else:
            frames = range(N_frames)

        ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)

        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=30, metadata=dict(artist="Me"), bitrate=1800)

        ani.save(path + "epi_opt.mp4", writer=writer)
        return None

    def test_convergence(self, R_means, alpha):
        """Tests convergence of EPI constraints.

        :param R_means: blah
        :type R_means: np.ndarray
        :param alpha: blah
        :type alpha: float
        """
        M, m = R_means.shape
        gt = np.sum(R_means > 0.0, axis=0).astype(np.float32)
        lt = np.sum(R_means < 0.0, axis=0).astype(np.float32)
        p_vals = 2 * np.minimum(gt / M, lt / M)
        return np.prod(p_vals > (alpha / m))

    def _opt_it_df(self, k, iter, H, R, R_keys):
        d = {"k": k, "iteration": iter, "H": H, "converged":None}
        d.update(zip(R_keys, list(R)))
        return pd.DataFrame(d, index=[0])

    def _save_epi_opt(self, save_path, opt_df, etas, cs):
        np.savez(save_path + "opt_data.npz", etas=etas, cs=cs)
        opt_df.to_csv(save_path + "opt_data.csv")

    def get_save_path(self, mu, arch, AL_hps, eps_name=None):
        epi_path = self.get_epi_path(mu, eps_name=eps_name)
        arch_string = arch.to_string()
        hp_string = AL_hps.to_string()
        return epi_path + "/%s/" % arch_string
        #return epi_path + "/%s_%s/" % (
        #    arch_string,
        #    hp_string,
        #)

    def get_epi_path(self, mu, eps_name=None):
        if eps_name is not None:
            _eps_name = eps_name
        else:
            if self.eps is not None:
                _eps_name = self.eps.__name__
            else:
                raise AttributeError("Model.eps is not set.")
        mu_string = array_str(mu)
        #return "data/%s_%s_mu=%s/" % (
        return "data/%s_%s/" % (
            self.name,
            _eps_name,
            #mu_string,
        )


    def load_epi_dist(
        self,
        mu,
        k=None,
        alpha=None,
        nu=0.1,
        arch_type="autoregressive",
        num_stages=1,
        num_layers=2,
        num_units=None,
        batch_norm=True,
        bn_momentum=0.99,
        post_affine=True,
        random_seed=1,
        init_type="iso_gauss",
        init_params={"loc": 0.0, "scale": 1.0},
        N=500,
        lr=1e-3,
        c0=1.0,
        gamma=0.25,
        beta=4.0,
    ):

        if k is not None:
            if type(k) is not int:
                raise TypeError(format_type_err_msg("Model.load_epi_dist", "k", k, int))
            if k < 0:
                raise ValueError("k must be augmented Lagrangian iteration index.")

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

        aug_lag_hps = AugLagHPs(N, lr, c0, gamma, beta)
        optimizer = tf.keras.optimizers.Adam(lr)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=nf)
        ckpt_dir = self.get_save_path(mu, nf, aug_lag_hps)
        ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt_state is not None:
            ckpts = ckpt_state.all_model_checkpoint_paths
        else:
            raise ValueError("No checkpoints found.")

        if k is not None:
            if k >= len(ckpts):
                raise ValueError("Index of checkpoint 'k' too large.")
            status = checkpoint.restore(ckpts[k])
            status.expect_partial()
            q_theta = Distribution(nf, self.parameters)
            return q_theta

    def parameter_check(self, parameters, verbose=False):
        """Check that model parameter list has no duplicates and valid bounds.

        :param parameters: List of :obj:`epi.models.Parameter`s.
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

            bounds = param.bounds
            if bounds[0] == bounds[1]:
                if verbose:
                    print(
                        "Warning: Left bound equal to right bound for parameter %s."
                        % name
                    )
                return False
            elif bounds[0] > bounds[1]:
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

    :param parameters: List of :obj:`epi.models.Parameter`.
    :type parameters: list
    :param nf: Normalizing flow trained via tensorflow.
    :type nf: :obj:`epi.normalizing_flows.NormalizingFlow`.
    """

    def __init__(self, nf, parameters=None):
        self._set_nf(nf)
        self.D = self.nf.D
        self._set_parameters(parameters)

    def __call__(self, N):
        z, _ = self.nf(N)
        return z.numpy()

    def _set_nf(self, nf):
        #if type(nf) is not NormalizingFlow:
        #    raise TypeError(format_type_err_msg(self, nf, "nf", NormalizingFlow))
        self.nf = nf

    def _set_parameters(self, parameters):
        if parameters is None:
            parameters = [Parameter("z%d" % (i + 1)) for i in range(self.D)]
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

    def plot_dist(self, N=200):
        z = self.sample(N)
        log_q_z = self.log_prob(z)
        df = pd.DataFrame(z)
        z_labels = [param.name for param in self.parameters]
        df.columns = z_labels
        df["log_q_z"] = log_q_z

        log_q_z_std = log_q_z - np.min(log_q_z)
        log_q_z_std = log_q_z_std / np.max(log_q_z_std)
        cmap = plt.get_cmap("viridis")
        g = sns.PairGrid(df, vars=z_labels)
        g = g.map_diag(sns.kdeplot)
        g = g.map_upper(plt.scatter, color=cmap(log_q_z_std))

        g = g.map_lower(sns.kdeplot)
        plt.show()
        return g


def format_opt_msg(k, i, cost, H, R):
    s1 = "" if cost < 0.0 else " "
    s2 = "" if H < 0.0 else " "
    args = (k, i, s1, cost, s2, H, np.sum(np.square(R)))
    return "EPI(k=%2d,i=%4d): cost %s%.2E, H %s%.2E, |R|^2 %.2E" % args
