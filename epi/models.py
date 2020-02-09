""" Models. """

import numpy as np
import inspect
import tensorflow as tf
from scipy.stats import ttest_ind
from epi.error_formatters import format_type_err_msg
from epi.normalizing_flows import NormalizingFlow
from epi.util import (
    gaussian_backward_mapping,
    aug_lag_vars,
    unbiased_aug_grad,
    AugLagHPs,
    array_str,
    np_column_vec,
)
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import seaborn as sns
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

    def set_eps(self, eps, m):
        """Set the emergent property statistic calculation for this model.

        The arguments of eps should be batch vectors of univariate parameter
        tensors following the naming convention in :obj:`self.Parameters`.

        :param eps: Emergent property statistics function.
        :type eps: function
        :param m: Dimensionality of emergent property statistics.
        :type m: int
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
        if len(T_z.shape) > 2:
            raise ValueError("Method eps must return tf.Tensor of dimension (N, D).")
        self.m = m
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
        num_iters=5000,
        N=500,
        lr=1e-3,
        c0=1.0,
        gamma=0.25,
        beta=4.0,
        alpha=0.05,
        nu=0.1,
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
        :param post_affine: Shift and scale following main transform.
        :type post_affine: bool, optional
        :param bounds: Bounds of distribution support, defaults to None.
        :type bounds: (np.ndarray, np.ndarray), optional
        :param random_seed: Random seed of architecture parameters, defaults to 1.
        :type random_seed: int, optional
        :param init_type: :math:`\\in` :obj:`['iso_gauss']`.
        :type init_type: str
        :param init_params: Parameters according to :obj:`init_type`.
        :type init_params: dict
        :param K: Number of augmented Lagrangian iterations
        :type K: int
        :param num_iters: Number of optimization iterations, Defaults to 500.
        :type num_iters: int, optional
        :param N: Number of batch samples per iteration.
        :type N: int
        :param lr: Adam optimizer learning rate, defaults to 1e-3.
        :type lr: float, optional
        :param alpha: P-value threshold for convergence testing, defaults to 0.05.
        :type alpha: float, optional
        :param nu: Fraction of N for convergence testing, defaults to 0.1.
        :type nu: float, optional
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
        print("Initializing architecture..")
        nf.initialize(init_type, init_params)
        print("done")

        # Checkpoint the initialization.
        optimizer = tf.keras.optimizers.Adam(lr)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=nf)
        ckpt_dir = self.get_save_path(mu, nf, aug_lag_hps)
        manager = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=None)
        manager.save(checkpoint_number=0)

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
            return cost, H, R, z

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
        opt_it_dfs = [self.opt_it_df(0, 0, H_0.numpy(), R_0.numpy(), R_keys)]

        # Record samples for movie.
        if save_movie_data:
            zs = [z.numpy()]

        # Measure initial R norm distribution.
        mu_colvec = np_column_vec(mu).astype(np.float32).T
        norms = get_R_norm_dist(nf, self.eps, mu_colvec, M_norm, N)

        # EPI optimization
        print(format_opt_msg(0, 0, cost_0, H_0, R_0))
        for k in range(1, K + 1):
            etas[k - 1], cs[k - 1], eta, c
            for i in range(1, num_iters + 1):
                cost, H, R, z = train_step(eta, c)
                if i % log_rate == 0:
                    if verbose:
                        print(format_opt_msg(k, i, cost, H, R))
                    iter = (k - 1) * num_iters + i
                    opt_it_dfs.append(
                        self.opt_it_df(k, iter, H.numpy(), R.numpy(), R_keys)
                    )
                    if save_movie_data:
                        zs.append(z.numpy())
            if not verbose:
                print(format_opt_msg(k, i, cost, H, R))

            # Save epi optimization data following aug lag iteration k.
            opt_it_dfs = [pd.concat(opt_it_dfs, ignore_index=True)]
            self.save_epi_opt(mu, nf, aug_lag_hps, opt_it_dfs[0], cs, etas)
            manager.save(checkpoint_number=k)

            if k < K:
                # Check for convergence if early stopping.
                R_means = get_R_mean_dist(nf, self.eps, mu_colvec, M_test, N_test)
                if self.test_convergence(R_means.numpy(), alpha):
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
                ckpt_dir + "zs.npz",
                zs=np.array(zs),
                iters=np.arange(0, k * num_iters + 1, log_rate),
            )
        q_theta = Distribution(nf, self.parameters)
        return q_theta, opt_it_dfs[0]

    def test_convergence(self, R_means, alpha):
        M, m = R_means.shape
        gt = np.sum(R_means > 0.0, axis=0).astype(np.float32)
        lt = np.sum(R_means < 0.0, axis=0).astype(np.float32)
        p_vals = 2 * np.minimum(gt / M, lt / M)
        return np.prod(p_vals > (alpha / m))

    def opt_it_df(self, k, iter, H, R, R_keys):
        d = {"k": k, "iteration": iter, "H": H}
        d.update(zip(R_keys, list(R)))
        return pd.DataFrame(d, index=[0])

    def save_epi_opt(self, mu, arch, aug_lag_hps, opt_df, etas, cs):
        save_path = self.get_save_path(mu, arch, aug_lag_hps)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez(save_path + "opt_data.npz", etas=etas, cs=cs)
        opt_df.to_csv(save_path + "opt_data.csv")

    def get_save_path(self, mu, arch, AL_hps, eps_name=None):
        if eps_name is not None:
            _eps_name = eps_name
        else:
            if self.eps is not None:
                _eps_name = self.eps.__name__
            else:
                raise AttributeError("Model.eps is not set.")
        mu_string = array_str(mu)
        arch_string = arch.to_string()
        hp_string = AL_hps.to_string()
        return "data/%s_%s_mu=%s/%s_%s/" % (
            self.name,
            _eps_name,
            mu_string,
            arch_string,
            hp_string,
        )

    def epi_opt_movie(self, path):
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=30, metadata=dict(artist="Me"), bitrate=1800)
        return None

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
            q_theta = Distribution(self.params, nf)
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
        if type(nf) is not NormalizingFlow:
            raise TypeError(format_type_err_msg(self, nf, "nf", NormalizingFlow))
        self.nf = nf

    def _set_parameters(self, parameters):
        if parameters is None:
            parameters = [Parameter("z%d" % (i + 1)) for i in range(self.D)]
            self.parameters = parameters
        elif type(parameters) is not list:
            raise TypeError(format_type_err_msg(self, parameters, "parameters", list))
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


def format_opt_msg(k, i, cost, H, R):
    s1 = "" if cost < 0.0 else " "
    s2 = "" if H < 0.0 else " "
    args = (k, i, s1, cost, s2, H, np.sum(np.square(R)))
    return "EPI(k=%2d,i=%4d): cost %s%.2E, H %s%.2E, |R|^2 %.2E" % args
