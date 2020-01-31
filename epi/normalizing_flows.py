""" Normalizing flow architecture class definitions for param distributions. """

import numpy as np
import scipy.stats
import os
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

from epi.error_formatters import format_type_err_msg
from epi.util import (
    gaussian_backward_mapping,
    np_column_vec,
    save_tf_model,
    load_tf_model,
    init_path,
    array_str,
)

DTYPE = tf.float32
EPS = 1e-6


class Architecture:
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
    :type bn_momentrum: float
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
        batch_norm=True,
        bn_momentum=0.99,
        post_affine=True,
        bounds=None,
        random_seed=1,
    ):
        """Constructor method."""
        self._set_arch_type(arch_type)
        self._set_D(D)
        self._set_num_stages(num_stages)
        self._set_num_layers(num_layers)
        self._set_num_units(num_units)
        self._set_batch_norm(batch_norm)
        if (self.batch_norm):
            self._set_bn_momentum(bn_momentum)
        else:
            self.bn_momentum=None
        self._set_post_affine(post_affine)
        self._set_bounds(bounds)
        self._set_random_seed(random_seed)
        self.trainable_variables = []

        self.stages = []
        self.permutations = []
        if self.batch_norm:
            self.batch_norms = []

        tf.keras.backend.clear_session()

        self.q0 = tfd.MultivariateNormalDiag(loc=self.D * [0.0])

        np.random.seed(self.random_seed)
        for i in range(num_stages):
            if arch_type == "coupling":
                shift_and_log_scale_fn = tfb.real_nvp_default_template(
                    hidden_layers=num_layers * [num_units]
                )

                stage = tfb.RealNVP(
                    num_masked=self.D // 2,
                    shift_and_log_scale_fn=shift_and_log_scale_fn,
                )
            elif arch_type == "autoregressive":
                shift_and_log_scale_fn = tfb.AutoregressiveNetwork(
                    params=2, hidden_units=num_layers * [num_units]
                )
                stage = tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=shift_and_log_scale_fn
                )

            self.stages.append(stage)

            if i < self.num_stages - 1:
                self.permutations.append(tfb.Permute(np.random.permutation(self.D)))
                if self.batch_norm:
                    bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
                    self.batch_norms.append(tfb.BatchNormalization(batchnorm_layer=bn))

        if self.post_affine:
            self.scale = tfb.Scale(
                scale=tf.Variable(initial_value=tf.ones((D,)), name="a")
            )
            self.shift = tfb.Shift(
                shift=tf.Variable(initial_value=tf.zeros((D,)), name="b")
            )
            self.PA = tfb.Chain([self.shift, self.scale])

        if self.lb is not None and self.ub is not None:
            self.support_mapping = IntervalFlow(self.lb, self.ub)

    @tf.function
    def __call__(self, N):
        tf.random.set_seed(self.random_seed)

        x = self.q0.sample(N)
        log_q0 = self.q0.log_prob(x)

        self.all_transforms = []
        sum_ldj = 0.0
        for i in range(self.num_stages):
            stage_i = self.stages[i]
            sum_ldj += stage_i.forward_log_det_jacobian(x, event_ndims=1)
            x = stage_i(x)
            self.all_transforms.append(stage_i)
            for var in stage_i.trainable_variables:
                self.trainable_variables.append(var)
            if i < self.num_stages - 1:
                if self.batch_norm:
                    batch_norm_i = self.batch_norms[i]
                    sum_ldj += batch_norm_i.forward_log_det_jacobian(x, event_ndims=1)
                    x = batch_norm_i(x)
                    self.all_transforms.append(batch_norm_i)
                permutation_i = self.permutations[i]
                x = permutation_i(x)
                self.all_transforms.append(permutation_i)

        if self.post_affine:
            sum_ldj += self.PA.forward_log_det_jacobian(x, event_ndims=1)
            x = self.PA(x)
            self.all_transforms.append(self.PA)
            self.trainable_variables += self.PA.trainable_variables

        if self.lb is not None and self.ub is not None:
            _ldj, x = self.support_mapping.forward_log_det_jacobian(x)
            sum_ldj += _ldj

        log_q_x = log_q0 - sum_ldj
        return x, log_q_x

    def sample(self, N):
        """Generate N samples from the network.

        :param N: Number of samples.
        :type N: int
        :return: N samples and log determinant of the jacobians.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        self.__call__(N)

    def _set_arch_type(self, arch_type):  # Make this noninherited?
        arch_types = ["coupling", "autoregressive"]
        if type(arch_type) is not str:
            raise TypeError(format_type_err_msg(self, "arch_type", arch_type, str))
        if arch_type not in arch_types:
            raise ValueError(
                'Architecture arch_type must be "coupling" or "autoregressive"'
            )
        self.arch_type = arch_type

    def _set_D(self, D):
        if type(D) is not int:
            raise TypeError(format_type_err_msg(self, "D", D, int))
        elif D < 2:
            raise ValueError("CouplingArch D %d must be greater than 0." % D)
        self.D = D

    def _set_num_stages(self, num_stages):
        if type(num_stages) is not int:
            raise TypeError(format_type_err_msg(self, "num_stages", num_stages, int))
        elif num_stages < 1:
            raise ValueError("Arch stages %d must be greater than 0." % num_stages)
        self.num_stages = num_stages

    def _set_num_layers(self, num_layers):
        if type(num_layers) is not int:
            raise TypeError(format_type_err_msg(self, "num_layers", num_layers, int))
        elif num_layers < 1:
            raise ValueError(
                "Arch num_layers arg %d must be greater than 0." % num_layers
            )
        self.num_layers = num_layers

    def _set_num_units(self, num_units):
        if type(num_units) is not int:
            raise TypeError(format_type_err_msg(self, "num_units", num_units, int))
        elif num_units < 1:
            raise ValueError("Arch num_units %d must be greater than 0." % num_units)
        self.num_units = num_units

    def _set_batch_norm(self, batch_norm):
        if type(batch_norm) is not bool:
            raise TypeError(format_type_err_msg(self, "batch_norm", batch_norm, bool))
        self.batch_norm = batch_norm

    def _set_bn_momentum(self, bn_momentum):
        if type(bn_momentum) is not float:
            raise TypeError(format_type_err_msg(self, "bn_momentum", bn_momentum, float))
        self.bn_momentum = bn_momentum

    def _set_post_affine(self, post_affine):
        if type(post_affine) is not bool:
            raise TypeError(format_type_err_msg(self, "post_affine", post_affine, bool))
        self.post_affine = post_affine

    def _set_random_seed(self, random_seed):
        if type(random_seed) is not int:
            raise TypeError(format_type_err_msg(self, "random_seed", random_seed, int))
        self.random_seed = random_seed

    def _set_bounds(self, bounds):
        if bounds is not None:
            _type = type(bounds)
            if _type in [list, tuple]:
                len_bounds = len(bounds)
                if _type is list:
                    bounds = tuple(bounds)
            else:
                raise TypeError(
                    "Architecture argument bounds must be tuple or list not %s."
                    % _type.__name__
                )

            if len_bounds != 2:
                raise ValueError("Architecture bounds arg must be length 2.")

            for i, bound in enumerate(bounds):
                if type(bound) is not np.ndarray:
                    raise TypeError(
                        format_type_err_msg(self, "bounds[%d]" % i, bound, np.ndarray)
                    )

            self.lb, self.ub = bounds[0], bounds[1]
        else:
            self.lb, self.ub = None, None

    def initialize(
        self,
        init_type,
        init_params,
        N=500,
        num_iters=int(1e4),
        lr=1e-3,
        KL_th=None,
        load_if_cached=True,
        verbose=False,
    ):
        """Initializes architecture to gaussian distribution via variational inference.

        :math:`\\underset{q_\\theta \\in Q}{\mathrm{arg max}} H(q_\\theta) + \\eta^\\top \\mathbb{E}_{z \\sim q_\\theta}[T(z)]`

        where :math:`\\eta` and :math:`T(z)` for a multivariate gaussian are:

        :math:`\\eta = \\begin{bmatrix} \\Sigma^{-1}\\mu \\\\ \\mathrm{vec} \\left( -\\frac{1}{2}\\Sigma^{-1} \\right) \\end{bmatrix}`
        :math:`T(z) = \\begin{bmatrix} z \\\\ \\mathrm{vec} \\left( zz^\\top \\right) \\end{bmatrix}`

        (Only implemented isotropic gaussian at the moment.)

        :obj:`init_type` should be

        'iso_gauss' with parameters
        * :obj:`init_params.loc` set to scalar mean of each variable.
        * :obj:`init_params.scale` set to scale of each variable.

        :param init_type: :math:`\\in` `['iso_gauss']`
        :type init_type: str
        :param init_params: Parameters according to :obj:`init_type`.
        :type init_params: dict
        :param N: Number of batch samples per iteration.
        :type N: int
        :param num_iters: Number of optimization iterations, Defaults to 500.
        :type num_iters: int, optional
        :param lr: Adam optimizer learning rate, defaults to 1e-3.
        :type lr: float, optional
        :param KL_th: Quit early if KL below this threshold, defaults to None.
        :type KL_th: float, optional
        :param load_if_cached: If initialization has been optimized before, load it, defaults to True.
        :type load_if_cached: bool, optional
        :param verbose: Print verbose output, defaults to False.
        :type verbose: bool, optional


        """

        _init_path = init_path(self.to_string(), init_type, init_params)
        if load_if_cached and os.path.exists(_init_path + ".p"):
            print("Loading variables from cached initialization.")
            _, _ = self(N)
            load_tf_model(_init_path, self.trainable_variables)
            return None

        if KL_th is None:
            KL_th = self.D * 0.001

        if init_type == "iso_gauss":
            loc = init_params["loc"]
            scale = init_params["scale"]
            mu = np.array(self.D * [loc])
            Sigma = scale * np.eye(self.D)

        p_target = scipy.stats.multivariate_normal(mu, Sigma)
        eta = gaussian_backward_mapping(mu, Sigma)

        optimizer = tf.keras.optimizers.Adam(lr)

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                x, log_q_x = self(N)
                xl = x[:, :, tf.newaxis]
                xr = x[:, tf.newaxis, :]
                xxT = tf.matmul(xl, xr)
                xxT_vec = tf.reshape(xxT, (N, self.D ** 2))
                T_x = tf.concat((x, xxT_vec), axis=1)
                E_T_x = tf.reduce_mean(T_x, axis=0)

                E_log_q_x = tf.reduce_mean(log_q_x)  # negative entropy
                loss = E_log_q_x + -tf.reduce_sum(eta * E_T_x)

            params = self.trainable_variables
            gradients = tape.gradient(loss, params)

            optimizer.apply_gradients(zip(gradients, params))
            return loss

        for i in range(num_iters):
            loss = train_step()
            if i % 100 == 0:
                x, log_q_x = self(N)
                KL = np.mean(log_q_x) - np.mean(p_target.logpdf(x))
                if verbose:
                    print(i, "loss", loss, "KL", KL)

                if np.isnan(loss):
                    print("Error. Loss is nan. Quitting.")
                    return None
                if not np.isfinite(loss):
                    print("Error. Loss is inf. Quitting.")
                    return None
                if KL < KL_th:
                    print("Finished initializing")
                    save_tf_model(_init_path, self.trainable_variables)
                    return None

        save_tf_model(_init_path, self.trainable_variables)
        print("Final KL to target after initialization optimization: %.2E." % KL)
        return None

    def to_string(self,):
        """Converts architecture to string for file saving.

        :return: A unique string for the architecture parameterization.
        :rtype: str
        """
        if self.arch_type == "coupling":
            arch_type_str = "C"
        elif self.arch_type == "autoregressive":
            arch_type_str = "AR"

        arch_string = "D%d_%s%d_L%d_U%d" % (
            self.D,
            arch_type_str,
            self.num_stages,
            self.num_layers,
            self.num_units,
        )

        if self.batch_norm:
            arch_string += "_bnmom=%.2E" % self.bn_momentum

        if self.post_affine:
            arch_string += "_PA"

        if self.lb is not None and self.ub is not None:
            arch_string += "_lb=%s_ub=%s" % (array_str(self.lb), array_str(self.ub))

        arch_string += "_rs%d" % self.random_seed
        return arch_string


class IntervalFlow(tfp.bijectors.Bijector):
    """Bijector maps from :math:`\\mathcal{R}^N` to an interval.

    :param lb: Lower bound. N values are numeric including float('-inf').
    :type lb: np.ndarray
    :param ub: Upper bound. N values are numeric including float('inf').
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
        tanh_flg, softplus_flg = self.D * [0], self.D * [0]
        tanh_m, tanh_c = self.D * [1.0], self.D * [0.0]
        softplus_m, softplus_c = self.D * [1.0], self.D * [0.0]
        for i in range(self.D):
            lb_i, ub_i = self.lb[i], self.ub[i]
            has_lb = not np.isneginf(lb_i)
            has_ub = not np.isposinf(ub_i)
            if has_lb and has_ub:
                tanh_flg[i] = 1
                tanh_m[i] = (ub_i - lb_i) / 2.0
                tanh_c[i] = (ub_i + lb_i) / 2.0
            elif has_lb:
                softplus_flg[i] = 1
                softplus_m[i] = 1.0
                softplus_c[i] = lb_i
            elif has_ub:
                softplus_flg[i] = 1
                softplus_m[i] = -1.0
                softplus_c[i] = ub_i

        self.tanh_flg = tf.constant(tanh_flg, dtype=DTYPE)
        self.softplus_flg = tf.constant(softplus_flg, dtype=DTYPE)
        self.tanh_m = tf.constant(tanh_m, dtype=DTYPE)
        self.tanh_c = tf.constant(tanh_c, dtype=DTYPE)
        self.softplus_m = tf.constant(softplus_m, dtype=DTYPE)
        self.softplus_c = tf.constant(softplus_c, dtype=DTYPE)

    def forward_log_det_jacobian(self, x):
        """Runs bijector forward and calculates log det jac of the function.

        :param x: Input tensor.
        :type x: tf.Tensor

        :returns: The forward pass and log determinant of the jacobian.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        ldj = 0.0
        # Tanh stage
        tanh_x = tf.tanh(x)
        out = tf.math.multiply(self.tanh_m, tanh_x) + self.tanh_c
        tanh_ldj = tf.reduce_sum(
            tf.multiply(
                self.tanh_flg,
                tf.math.log(self.tanh_m + EPS)
                + tf.math.log(1.0 - tf.square(tanh_x) + EPS),
            ),
            1,
        )
        ldj += tanh_ldj
        x = tf.multiply(self.tanh_flg, out) + tf.multiply(1 - self.tanh_flg, x)

        out = (
            tf.math.multiply(self.softplus_m, tf.math.softplus(x))
            + self.softplus_c
        )
        softplus_ldj = tf.reduce_sum(
            tf.math.multiply(
                self.softplus_flg,
                tf.math.log_sigmoid(x),
            ),
            1,
        )
        ldj += softplus_ldj

        x = tf.multiply(self.softplus_flg, out) + tf.multiply(1 - self.softplus_flg, x)
        return x, ldj
