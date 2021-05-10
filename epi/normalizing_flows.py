""" Normalizing flow architecture class definitions for param distributions. """

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import tensorshape_util

tfb = tfp.bijectors
tfd = tfp.distributions
from epi.batch_norm import BatchNormalization
import tensorflow.compat.v1 as tf1
import pandas as pd

from epi.error_formatters import format_type_err_msg
from epi.util import (
    gaussian_backward_mapping,
    np_column_vec,
    get_hash,
    set_dir_index,
    array_str,
    dbg_check,
)

DTYPE = tf.float32
EPS = 1e-6


class NormalizingFlow(tf.keras.Model):
    """Normalizing flow network for approximating parameter distributions.

    The normalizing flow is constructed via stage(s) of either coupling or
    autoregressive transforms of :math:`q_0`.  Coupling transforms are real NVP bijectors
    where each conditional distribution has the same number of neural network
    layers and units.  One stage is one coupling (second half of elements are
    conditioned on the first half (see :obj:`tfp.bijectors.RealNVP`)). Similarly, 
    autoregressive transforms are masked autoregressive flow (MAF) bijectors. One stage
    is one full autoregressive factorization (see :obj:`tfp.bijectors.MAF`).

    After each stage, which is succeeded by another coupling or autoregressive 
    transform, the dimensions are permuted via a :obj:`tfp.bijectors.Permute` bijector 
    followed by a :obj:`tfp.bijectors.BatchNormalization`
    bijector.  This facilitates randomized conditioning (real NVP) and
    factorization orderings (MAF) at each stage.

    E.g. :obj:`arch_type='autoregressive', num_stages=2`

    :math:`q_0` -> MAF -> permute -> batch norm -> MAF -> ...

    We parameterize the final processing stages of the normalizing flow (a deep 
    generative model) via post_affine and bounds.

    To facilitate scaling and shifting of the normalizing flow up to this point,
    one can set post_affine to True.

    E.g. :obj:`arch_type='autoregressive', num_stages=2, post_affine=True`

    :math:`q_0` -> MAF -> permute -> batch norm -> MAF -> PA -> ...

    By setting bounds to a tuple (lower_bound, upper_bound), the final step
    in the normalizing flow maps to the support of the distribution using an
    :obj:`epi.normalizing_flows.IntervalFlow`.

    E.g. :obj:`arch_type='autoregressive', num_stages=2, post_affine=True, bounds=(lb,ub)`

    :math:`q_0` -> MAF -> permute -> batch norm -> MAF -> post affine -> interval flow

    The base distribution :math:`q_0` is chosen to be a standard isotoropic gaussian.

    :param arch_type: :math:`\\in` `['autoregressive', 'coupling']`
    :type arch_type: str
    :param D: Dimensionality of the normalizing flow.
    :type D: int
    :param num_stages: Number of coupling or autoregressive stages.
    :type num_stages: int
    :param num_layers: Number of neural network layer per conditional.
    :type num_layers: int
    :param num_units: Number of units per layer.
    :type num_units: int
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
    """

    def __init__(
        self,
        arch_type,
        D,
        num_stages,
        num_layers,
        num_units,
        elemwise_fn="affine",
        num_bins=32,
        batch_norm=True,
        bn_momentum=0.0,
        post_affine=True,
        bounds=None,
        random_seed=1,
    ):
        """Constructor method."""
        super(NormalizingFlow, self).__init__()
        self._set_arch_type(arch_type)
        self._set_D(D)
        self._set_num_stages(num_stages)
        self._set_num_layers(num_layers)
        self._set_num_units(num_units)
        self._set_elemwise_fn(elemwise_fn)
        if (self.arch_type == "autoregressive" and self.elemwise_fn == "spline"):
            raise NotImplementedError("Error: MAF flows with splines are not implemented yet.")
        self._set_num_bins(num_bins)
        self._set_batch_norm(batch_norm)
        if not self.batch_norm:
            self.bn_momentum = None
        self._set_post_affine(post_affine)
        self._set_bounds(bounds)
        self._set_random_seed(random_seed)

        self.stages = []
        # self.shift_and_log_scale_fns = []
        self.bijector_fns = []
        self.permutations = []
        if self.batch_norm:
            self.batch_norms = []

        self.q0 = tfd.MultivariateNormalDiag(loc=self.D * [0.0])
        bijectors = []

        np.random.seed(self.random_seed)
        for i in range(num_stages):
            if arch_type == "coupling":
                num_masked = self.D // 2
                if elemwise_fn == "affine":
                    bijector_fn = tfb.real_nvp_default_template(
                        hidden_layers=num_layers * [num_units]
                    )
                    stage = tfb.RealNVP(
                        num_masked=num_masked, shift_and_log_scale_fn=bijector_fn
                    )
                    self.bijector_fns.append(bijector_fn)
                elif elemwise_fn == "spline":
                    bijector_fn = SplineParams(D - num_masked, num_layers, num_units,
                                               self.num_bins, B=4.)
                    stage = tfb.RealNVP(num_masked=num_masked, bijector_fn=bijector_fn)
                    # make parameters visible to autograd
                    self.bijector_fns.append(bijector_fn._bin_widths)
                    self.bijector_fns.append(bijector_fn._bin_heights)
                    self.bijector_fns.append(bijector_fn._knot_slopes)

            elif arch_type == "autoregressive":
                # Splines are not implemented for MAF networks yet!
                bijector_fn = tfb.AutoregressiveNetwork(
                    params=2, hidden_units=num_layers * [num_units]
                )
                stage = tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=bijector_fn
                )
                self.bijector_fns.append(bijector_fn)

            self.stages.append(stage)
            bijectors.append(stage)
            #self.shift_and_log_scale_fns.append(shift_and_log_scale_fn)
            self.bijector_fns.append(bijector_fn)

            if i < self.num_stages - 1:
                lower_upper, perm = trainable_lu_factorization(self.D)
                perm_i = tfp.bijectors.ScaleMatvecLU(
                    lower_upper, perm, validate_args=False, name=None
                )
                self.permutations.append(perm_i)
                bijectors.append(perm_i)
                if self.batch_norm:
                    bn = tf.keras.layers.BatchNormalization(momentum=bn_momentum)
                    batch_norm_i = BatchNormalization(batchnorm_layer=bn)
                    self.batch_norms.append(batch_norm_i)
                    bijectors.append(batch_norm_i)

        if self.post_affine:
            self.a = tf.Variable(initial_value=tf.ones((D,)), name="a")
            self.b = tf.Variable(initial_value=tf.zeros((D,)), name="b")
            self.scale = tfb.Scale(scale=self.a)
            self.shift = tfb.Shift(shift=self.b)
            self.PA = tfb.Chain([self.shift, self.scale])
            bijectors.append(self.PA)

        # TODO Make this "or" ?
        if self.lb is not None and self.ub is not None:
            self.support_mapping = IntervalFlow(self.lb, self.ub)
            bijectors.append(self.support_mapping)

        bijectors.reverse()
        self.trans_dist = tfd.TransformedDistribution(
            distribution=self.q0, bijector=tfb.Chain(bijectors)
        )

        if self.batch_norm:
            self._set_bn_momentum(bn_momentum)

    def __call__(self, N):
        x = self.q0.sample(N)
        log_q0 = self.q0.log_prob(x)

        sum_ldj = 0.0
        for i in range(self.num_stages):
            stage_i = self.stages[i]
            sum_ldj += stage_i.forward_log_det_jacobian(x, event_ndims=1)
            x = stage_i(x)
            if i < self.num_stages - 1:
                permutation_i = self.permutations[i]
                x = permutation_i(x)
                sum_ldj += permutation_i.forward_log_det_jacobian(x, event_ndims=1)
                if self.batch_norm:
                    batch_norm_i = self.batch_norms[i]
                    sum_ldj += batch_norm_i.forward_log_det_jacobian(x, event_ndims=1)
                    x = batch_norm_i(x)

        if self.post_affine:
            sum_ldj += self.PA.forward_log_det_jacobian(x, event_ndims=1)
            x = self.PA(x)

        if self.lb is not None and self.ub is not None:
            x, _ldj = self.support_mapping.forward_and_log_det_jacobian(x)
            sum_ldj += _ldj

        log_q_x = log_q0 - sum_ldj
        return x, log_q_x

    # @tf.function
    def sample(self, N):
        """Generate N samples from the network.

        :param N: Number of samples.
        :type N: int
        :return: N samples and log determinant of the jacobians.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        return self.__call__(N)[0]

    def _set_arch_type(self, arch_type):  # Make this noninherited?
        arch_types = ["coupling", "autoregressive"]
        if type(arch_type) is not str:
            raise TypeError(format_type_err_msg(self, "arch_type", arch_type, str))
        if arch_type not in arch_types:
            raise ValueError(
                'NormalizingFlow arch_type must be "coupling" or "autoregressive"'
            )
        self.arch_type = arch_type

    def _set_D(self, D):
        if type(D) is not int:
            raise TypeError(format_type_err_msg(self, "D", D, int))
        elif D < 2:
            raise ValueError("NormalizingFlow D %d must be greater than 0." % D)
        self.D = D

    def _set_num_stages(self, num_stages):
        if type(num_stages) is not int:
            raise TypeError(format_type_err_msg(self, "num_stages", num_stages, int))
        elif num_stages < 0:
            raise ValueError(
                "NormalizingFlow num_stages %d must be greater than 0." % num_stages
            )
        self.num_stages = num_stages

    def _set_num_layers(self, num_layers):
        if type(num_layers) is not int:
            raise TypeError(format_type_err_msg(self, "num_layers", num_layers, int))
        elif num_layers < 1:
            raise ValueError(
                "NormalizingFlow num_layers arg %d must be greater than 0." % num_layers
            )
        self.num_layers = num_layers

    def _set_num_units(self, num_units):
        if type(num_units) is not int:
            raise TypeError(format_type_err_msg(self, "num_units", num_units, int))
        elif num_units < 1:
            raise ValueError(
                "NormalizingFlow num_units %d must be greater than 0." % num_units
            )
        self.num_units = num_units

    def _set_elemwise_fn(self, elemwise_fn):
        elemwise_fns = ["affine", "spline"]
        if type(elemwise_fn) is not str:
            raise TypeError(format_type_err_msg(self, "elemwise_fn", elemwise_fn, str))
        if elemwise_fn not in elemwise_fns:
            raise ValueError('NormalizingFlow elemwise_fn must be "affine" or "spline"')
        self.elemwise_fn = elemwise_fn

    def _set_num_bins(self, num_bins):
        if type(num_bins) is not int:
            raise TypeError(format_type_err_msg(self, "num_bins", num_bins, int))
        elif num_bins < 2:
            raise ValueError(
                "NormalizingFlow spline num_bins %d must be greater than 1." % num_units
            )
        self.num_bins = num_bins

    def _set_batch_norm(self, batch_norm):
        if type(batch_norm) is not bool:
            raise TypeError(format_type_err_msg(self, "batch_norm", batch_norm, bool))
        self.batch_norm = batch_norm

    def _set_bn_momentum(self, bn_momentum):
        if type(bn_momentum) is not float:
            raise TypeError(
                format_type_err_msg(self, "bn_momentum", bn_momentum, float)
            )
        self.bn_momentum = bn_momentum
        bijectors = self.trans_dist.bijector.bijectors
        for bijector in bijectors:
            if type(bijector).__name__ == "BatchNormalization":
                bijector.batchnorm.momentum = bn_momentum
        return None

    def _reset_bn_movings(self,):
        bijectors = self.trans_dist.bijector.bijectors
        for bijector in bijectors:
            if type(bijector).__name__ == "BatchNormalization":
                bijector.batchnorm.moving_mean.assign(np.zeros((self.D,)))
                bijector.batchnorm.moving_variance.assign(np.ones((self.D,)))
        return None

    def _set_post_affine(self, post_affine):
        if type(post_affine) is not bool:
            raise TypeError(format_type_err_msg(self, "post_affine", post_affine, bool))
        self.post_affine = post_affine

    def _set_bounds(self, bounds):
        if bounds is not None:
            _type = type(bounds)
            if _type in [list, tuple]:
                len_bounds = len(bounds)
                if _type is list:
                    bounds = tuple(bounds)
            else:
                raise TypeError(
                    "NormalizingFlow argument bounds must be tuple or list not %s."
                    % _type.__name__
                )

            if len_bounds != 2:
                raise ValueError("NormalizingFlow bounds arg must be length 2.")

            for i, bound in enumerate(bounds):
                if type(bound) is not np.ndarray:
                    raise TypeError(
                        format_type_err_msg(self, "bounds[%d]" % i, bound, np.ndarray)
                    )

            self.lb, self.ub = bounds[0], bounds[1]
        else:
            self.lb, self.ub = None, None

    def _set_random_seed(self, random_seed):
        if type(random_seed) is not int:
            raise TypeError(format_type_err_msg(self, "random_seed", random_seed, int))
        self.random_seed = random_seed

    def get_init_path(self, mu, Sigma):
        init_hash = get_hash([mu, Sigma, self.lb, self.ub])
        init_path = "./data/inits/%s/%s/" % (init_hash, self.to_string())
        if not os.path.exists(init_path):
            os.makedirs(init_path)

        init_index = {"mu": mu, "Sigma": Sigma, "lb": self.lb, "ub": self.ub}
        init_index_file = "./data/inits/%s/init.pkl" % init_hash

        arch_index = {
            "arch_type": self.arch_type,
            "D": self.D,
            "num_stages": self.num_stages,
            "num_layers": self.num_layers,
            "num_units": self.num_units,
            "elemwise_fn": self.elemwise_fn,
            "num_bins": self.num_bins,
            "batch_norm": self.batch_norm,
            "bn_momentum": self.bn_momentum,
            "post_affine": self.post_affine,
            "lb": self.lb,
            "ub": self.ub,
            "random_seed": self.random_seed,
        }
        arch_index_file = "./data/inits/%s/%s/arch.pkl" % (init_hash, self.to_string())

        set_dir_index(init_index, init_index_file)
        set_dir_index(arch_index, arch_index_file)

        return init_path

    def initialize(
        self,
        mu,
        Sigma,
        N=500,
        num_iters=int(1e4),
        lr=1e-3,
        log_rate=100,
        load_if_cached=True,
        save=True,
        verbose=False,
    ):
        """Initializes architecture to gaussian distribution via variational inference.

        :math:`\\underset{q_\\theta \\in Q}{\\mathrm{arg max}} H(q_\\theta) + \\eta^\\top \\mathbb{E}_{z \\sim q_\\theta}[T(z)]`

        where :math:`\\eta` and :math:`T(z)` for a multivariate gaussian are:

        :math:`\\eta = \\begin{bmatrix} \\Sigma^{-1}\\mu \\\\ \\mathrm{vec} \\left( -\\frac{1}{2}\\Sigma^{-1} \\right) \\end{bmatrix}`
        :math:`T(z) = \\begin{bmatrix} z \\\\ \\mathrm{vec} \\left( zz^\\top \\right) \\end{bmatrix}`

        Parameter `init_type` may be:

        :obj:`'iso_gauss'` with parameters

        * :obj:`init_params.loc` set to scalar mean of each variable.
        * :obj:`init_params.scale` set to scale of each variable.
        
        :obj:`'gaussian'` with parameters

        * :obj:`init_params.mu` set to the mean.
        * :obj:`init_params.Sigma` set to the covariance.

        :param init_type: :math:`\\in` `['iso_gauss', 'gaussian']`
        :type init_type: str
        :param init_params: Parameters according to :obj:`init_type`.
        :type init_params: dict
        :param N: Number of batch samples per iteration.
        :type N: int
        :param num_iters: Number of optimization iterations, defaults to 500.
        :type num_iters: int, optional
        :param lr: Adam optimizer learning rate, defaults to 1e-3.
        :type lr: float, optional
        :param log_rate: Record optimization data every so iterations, defaults to 100.
        :type log_rate: int, optional
        :param load_if_cached: If initialization has been optimized before, load it, defaults to True.
        :type load_if_cached: bool, optional
        :param save: Save initialization if true, defaults to True.
        :type save: bool, optional
        :param verbose: Print verbose output, defaults to False.
        :type verbose: bool, optional

        """
        optimizer = tf.keras.optimizers.Adam(lr)

        init_path = self.get_init_path(mu, Sigma)
        init_file = init_path + "ckpt"
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
        ckpt = tf.train.latest_checkpoint(init_path)
        if load_if_cached and (ckpt is not None):
            print("Loading variables from cached initialization.")
            status = checkpoint.restore(ckpt)
            status.expect_partial()  # Won't use optimizer momentum parameters
            opt_data_file = init_path + "opt_data.csv"
            if os.path.exists(opt_data_file):
                return pd.read_csv(opt_data_file)

        t1 = time.time()

        eta = gaussian_backward_mapping(mu, Sigma)

        def gauss_init_loss(z, log_q_z, eta):
            zl = z[:, :, tf.newaxis]
            zr = z[:, tf.newaxis, :]
            zzT = tf.matmul(zl, zr)
            zzT_vec = tf.reshape(zzT, (N, self.D ** 2))
            T_z = tf.concat((z, zzT_vec), axis=1)
            E_T_z = tf.reduce_mean(T_z, axis=0)

            E_log_q_z = tf.reduce_mean(log_q_z)
            loss = E_log_q_z - tf.reduce_sum(eta * E_T_z)
            return loss

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                z, log_q_z = self(N)
                loss = gauss_init_loss(z, log_q_z, eta)

            params = self.trainable_variables
            gradients = tape.gradient(loss, params)
            ming, maxg = -1e5, 1e5
            gradients = [tf.clip_by_value(grad, ming, maxg) for grad in gradients]

            optimizer.apply_gradients(zip(gradients, params))

            return loss

        z, log_q_z = self(N)
        loss0 = gauss_init_loss(z, log_q_z, eta).numpy()
        H0 = -np.mean(log_q_z.numpy())
        KL0 = self.gauss_KL(z, log_q_z, mu, Sigma)
        d = {"iteration": 0, "loss": loss0, "H": H0, "KL": KL0}
        opt_it_dfs = [pd.DataFrame(d, index=[0])]

        for i in range(1, num_iters + 1):
            start_time = time.time()
            loss = train_step()
            ts_time = time.time() - start_time
            if np.isnan(loss):
                raise ValueError("Initialization loss is nan.")
            if not np.isfinite(loss):
                raise ValueError("Initialization loss is inf.")

            if i % log_rate == 0:
                z, log_q_z = self(N)
                loss = gauss_init_loss(z, log_q_z, eta).numpy()
                H = -np.mean(log_q_z.numpy())
                KL = self.gauss_KL(z, log_q_z, mu, Sigma)
                d = {"iteration": i, "loss": loss, "H": H, "KL": KL}
                opt_it_dfs.append(pd.DataFrame(d, index=[0]))
                if verbose:
                    if not np.isnan(KL):
                        print(
                            i,
                            "H",
                            H,
                            "KL",
                            KL,
                            "loss",
                            loss,
                            "%.2E s" % ts_time,
                            flush=True,
                        )
                        else:
                        print(i, "H", H, "loss", loss, "%.2E s" % ts_time, flush=True)
        init_time = time.time() - t1
        opt_df = pd.concat(opt_it_dfs, ignore_index=True)
        opt_df.to_csv(init_path + "opt_data.csv")
        np.savez(os.path.join(init_path, "timing.npz"),
                 init_time=init_time)
        checkpoint.save(file_prefix=init_file)
        return opt_df

    def gauss_KL(self, z, log_q_z, mu, Sigma):
        if self.lb is not None or self.ub is not None:
            return np.nan
        q_true = scipy.stats.multivariate_normal(mean=mu, cov=Sigma)
        return np.mean(log_q_z) - np.mean(q_true.logpdf(z))

    def to_string(self,):
        """Converts architecture to string for file saving.

        :return: A unique string for the architecture parameterization.
        :rtype: str
        """
        if self.arch_type == "coupling":
            arch_type_str = "C"
        elif self.arch_type == "autoregressive":
            arch_type_str = "AR"

        arch_string = "D%d_%s%d_%s_L%d_U%d" % (
            self.D,
            arch_type_str,
            self.num_stages,
            self.elemwise_fn,
            self.num_layers,
            self.num_units,
        )

        if self.elemwise_fn == "spline":
            arch_string += "_bins=%d" % self.num_bins

        if self.batch_norm:
            arch_string += "_bnmom=%.2E" % self.bn_momentum

        if self.post_affine:
            arch_string += "_PA"

        # if self.lb is not None and self.ub is not None:
        # arch_string += "_lb=%s_ub=%s" % (array_str(self.lb), array_str(self.ub))

        arch_string += "_rs%d" % self.random_seed
        return arch_string


class IntervalFlow(tfp.bijectors.Bijector):
    """Bijector maps from :math:`\\mathcal{R}^N` to an interval.

    Each dimension is handled independently according to the type of bound.

    * no bound: :math:`y_i = x_i`
    * only lower bound: :math:`y_i = \\log(1 + \\exp(x_i)) + lb_i`
    * only upper bound: :math:`y_i = -\\log(1 + \\exp(x_i)) + ub_i`
    * upper and lower bound: :math:`y_i = \\frac{ub_i - lb_i} \\sigmoid(x_i) + \\frac{ub_i + lb_i}{2}`

    :param lb: Lower bound. N values are numeric including :obj:`float('-inf')`.
    :type lb: np.ndarray
    :param ub: Upper bound. N values are numeric including :obj:`float('inf')`.
    :type ub: np.ndarray
    """

    def __init__(self, lb, ub):
        """Constructor method."""
        super().__init__(forward_min_event_ndims=1, inverse_min_event_ndims=1)
        # Check types.
        if type(lb) not in [list, np.ndarray]:
            raise TypeError(format_type_err_msg(self, "lb", lb, np.ndarray))
        if type(ub) not in [list, np.ndarray]:
            raise TypeError(format_type_err_msg(self, "ub", ub, np.ndarray))

        # Handle list input.
        if type(lb) is list:
            lb = np.array(lb)
        if type(ub) is list:
            ub = np.array(ub)

        # Make sure we have 1-D np vec
        self.lb = np_column_vec(lb)[:, 0]
        self.ub = np_column_vec(ub)[:, 0]

        if self.lb.shape[0] != self.ub.shape[0]:
            raise ValueError("lb and ub have different lengths.")
        self.D = self.lb.shape[0]

        for lb_i, ub_i in zip(self.lb, self.ub):
            if lb_i >= ub_i:
                raise ValueError("Lower bound %.2E > upper bound %.2E." % (lb_i, ub_i))
        sigmoid_flg, softplus_flg = self.D * [0], self.D * [0]
        sigmoid_m, sigmoid_c = self.D * [1.0], self.D * [0.0]
        softplus_m, softplus_c = self.D * [1.0], self.D * [0.0]
        for i in range(self.D):
            lb_i, ub_i = self.lb[i], self.ub[i]
            has_lb = not np.isneginf(lb_i)
            has_ub = not np.isposinf(ub_i)
            if has_lb and has_ub:
                sigmoid_flg[i] = 1
                sigmoid_m[i] = ub_i - lb_i
                sigmoid_c[i] = lb_i
            elif has_lb:
                softplus_flg[i] = 1
                softplus_m[i] = 1.0
                softplus_c[i] = lb_i
            elif has_ub:
                softplus_flg[i] = 1
                softplus_m[i] = -1.0
                softplus_c[i] = ub_i

        self.sigmoid_flg = tf.constant(sigmoid_flg, dtype=DTYPE)
        self.softplus_flg = tf.constant(softplus_flg, dtype=DTYPE)
        self.sigmoid_m = tf.constant(sigmoid_m, dtype=DTYPE)
        self.sigmoid_c = tf.constant(sigmoid_c, dtype=DTYPE)
        self.softplus_m = tf.constant(softplus_m, dtype=DTYPE)
        self.softplus_c = tf.constant(softplus_c, dtype=DTYPE)

    def forward_and_log_det_jacobian(self, x):
        """Runs bijector forward and calculates log det jac of the function.

        It's more efficient to run samples and ldjs forward together for EPI.

        :param x: Input tensor.
        :type x: tf.Tensor

        :returns: The forward pass and log determinant of the jacobian.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        ldj = 0.0
        EPS_log = 1e-3
        sigmoid_x = tf.sigmoid(x)
        out = tf.math.multiply(self.sigmoid_m, sigmoid_x) + self.sigmoid_c
        sigmoid_ldj = tf.reduce_sum(
            tf.multiply(
                self.sigmoid_flg,
                tf.math.log(self.sigmoid_m)
                + tf.math.log_sigmoid(x)
                + tf.math.log_sigmoid(-x),
            ),
            1,
        )
        ldj += sigmoid_ldj
        x = tf.multiply(self.sigmoid_flg, out) + tf.multiply(1 - self.sigmoid_flg, x)

        out = tf.math.multiply(self.softplus_m, tf.math.softplus(x)) + self.softplus_c
        softplus_ldj = tf.reduce_sum(
            tf.math.multiply(self.softplus_flg, tf.math.log_sigmoid(x)), 1
        )
        ldj += softplus_ldj

        x = tf.multiply(self.softplus_flg, out) + tf.multiply(1 - self.softplus_flg, x)
        return x, ldj

    def forward(self, x):
        """Runs bijector forward and calculates log det jac of the function.

        :param x: Input tensor.
        :type x: tf.Tensor

        :returns: The forward pass of the interval flow
        :rtype: (tf.Tensor, tf.Tensor)
        """
        sigmoid_x = tf.math.sigmoid(x)
        out = tf.math.multiply(self.sigmoid_m, sigmoid_x) + self.sigmoid_c
        x = tf.multiply(self.sigmoid_flg, out) + tf.multiply(1 - self.sigmoid_flg, x)

        out = tf.math.multiply(self.softplus_m, tf.math.softplus(x)) + self.softplus_c
        x = tf.multiply(self.softplus_flg, out) + tf.multiply(1 - self.softplus_flg, x)
        return x

    def inverse(self, x):
        """Inverts bijector at value x.

        :param x: Input tensor.
        :type x: tf.Tensor

        :returns: The backward pass of the interval flow
        :rtype: (tf.Tensor, tf.Tensor)
        """
        softplus_inv = tf.math.log(
            tf.math.exp(
                tf.multiply(
                    self.softplus_flg, tf.divide(x - self.softplus_c, self.softplus_m)
                )
            )
            - 1
            + EPS
        )
        x = tf.multiply(self.softplus_flg, softplus_inv) + tf.multiply(
            1 - self.softplus_flg, x
        )

        logit_input = tf.multiply(
            self.sigmoid_flg, tf.divide(x - self.sigmoid_c, self.sigmoid_m + EPS)
        )
        logit = tf.math.log(logit_input + EPS) - tf.math.log(1.0 - logit_input + EPS)

        x = tf.multiply(self.sigmoid_flg, logit) + tf.multiply(
            1.0 - self.sigmoid_flg, x
        )
        return x

    def forward_log_det_jacobian(self, x):
        """Calculates forward log det jac of the interval flow.

        :param x: Input tensor.
        :type x: tf.Tensor

        :returns: Log determinant of the jacobian of interval flow.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        ldj = 0.0
        # Tanh stage
        sigmoid_x = tf.sigmoid(x)
        out = tf.math.multiply(self.sigmoid_m, sigmoid_x) + self.sigmoid_c
        sigmoid_ldj = tf.reduce_sum(
            tf.multiply(
                self.sigmoid_flg,
                tf.math.log(self.sigmoid_m)
                + tf.math.log_sigmoid(x)
                + tf.math.log_sigmoid(-x),
            ),
            1,
        )
        ldj += sigmoid_ldj
        x = tf.multiply(self.sigmoid_flg, out) + tf.multiply(1 - self.sigmoid_flg, x)

        softplus_ldj = tf.reduce_sum(
            tf.math.multiply(self.softplus_flg, tf.math.log_sigmoid(x)), 1
        )
        ldj += softplus_ldj

        return ldj

    def inverse_log_det_jacobian(self, x, event_ndims=1):
        """Log determinant jacobian of inverse pass.

        :param x: Input tensor.
        :type x: tf.Tensor

        :returns: The inverse log determinant jacobian.
        :rtype: (tf.Tensor, tf.Tensor)
        """

        ildj = -self.forward_log_det_jacobian(self.inverse(x))
        return ildj


def hp_df_to_nf(hp_df, model):
    nf = NormalizingFlow(
        arch_type=hp_df["arch_type"],
        D=model.D,
        num_stages=int(hp_df["num_stages"]),
        num_layers=int(hp_df["num_layers"]),
        num_units=int(hp_df["num_units"]),
        batch_norm=bool(hp_df["batch_norm"]),
        bn_momentum=float(hp_df["bn_momentum"]),
        post_affine=bool(hp_df["post_affine"]),
        bounds=model._get_bounds(),
        random_seed=int(hp_df["random_seed"]),
    )
    return nf


class SplineParams(tf.Module):
    def __init__(self, D, num_layers, num_units, nbins=32, B=1):
        self._D = D
        self._num_layers = num_layers
        self._num_units = num_units
        self._nbins = nbins
        self._built = False
        self._B = B

        self._bin_widths = tf.keras.Sequential()
        self._bin_heights = tf.keras.Sequential()
        self._knot_slopes = tf.keras.Sequential()

        for i in range(num_layers-1):
            self._bin_widths.add(tf.keras.layers.Dense(
                num_units, activation='tanh', name="w%d" % (i+1)
            ))
            self._bin_heights.add(tf.keras.layers.Dense(
                num_units, activation='tanh', name="h%d" % (i+1)
            ))
            self._knot_slopes.add(tf.keras.layers.Dense(
                num_units, activation='tanh', name="s%d" % (i+1)
            ))

        self._bin_widths.add(tf.keras.layers.Dense(
            D * self._nbins, activation=self._bin_positions, name="w%d" % (num_layers)
        ))
        self._bin_heights.add(tf.keras.layers.Dense(
            D * self._nbins, activation=self._bin_positions, name="h%d" % (num_layers)
        ))
        self._knot_slopes.add(tf.keras.layers.Dense(
            D * (self._nbins - 1), activation=self._slopes, name="s%d" % (num_layers)
        ))
        self._built = True

    def _bin_positions(self, x):
        x = tf.reshape(x, [-1, self._D, self._nbins])
        return tf.math.softmax(x, axis=-1) * (2 * self._B - self._nbins * 1e-2) + 1e-2

    def _slopes(self, x):
        x = tf.reshape(x, [-1, self._D, self._nbins - 1])
        return tf.math.softplus(x) + 1e-2

    def __call__(self, x, D):
        return tfb.RationalQuadraticSpline(
            bin_widths=self._bin_widths(x),
            bin_heights=self._bin_heights(x),
            knot_slopes=self._knot_slopes(x),
            range_min=-self._B,
        )


def trainable_lu_factorization(
    event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None
):
    with tf.name_scope(name or "trainable_lu_factorization"):
        event_size = tf.convert_to_tensor(
            event_size, dtype_hint=tf.int32, name="event_size"
        )
        batch_shape = tf.convert_to_tensor(
            batch_shape, dtype_hint=event_size.dtype, name="batch_shape"
        )
        random_matrix = tf.random.uniform(
            shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
            dtype=dtype,
            seed=seed,
        )
        random_orthonormal = tf.linalg.qr(random_matrix)[0]
        lower_upper, permutation = tf.linalg.lu(random_orthonormal)
        lower_upper = tf.Variable(
            initial_value=lower_upper, trainable=True, name="lower_upper"
        )
        # Initialize a non-trainable variable for the permutation indices so
        # that its value isn't re-sampled from run-to-run.
        permutation = tf.Variable(
            initial_value=permutation, trainable=False, name="permutation"
        )
    return lower_upper, permutation
