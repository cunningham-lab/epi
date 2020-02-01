""" Models. """

import numpy as np
import inspect
import tensorflow as tf
from epi.error_formatters import format_type_err_msg
from epi.normalizing_flows import Architecture 
from epi.util import gaussian_backward_mapping

REAL_NUMERIC_TYPES = (int, float)


class Parameter:
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


class Model:
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
        self.eps = lambda: None

    def _set_name(self, name):
        if type(name) is not str:
            raise TypeError(format_type_err_msg(self, "name", name, str))
        self.name = name

    def _set_parameters(self, parameters):
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

        z = tf.keras.Input(shape=(self.D))
        T_z = self.eps(z)
        if (len(T_z.shape) > 2):
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
        N=500,
        num_iters=10000,
        lr=1e-3,
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
        :param N: Number of batch samples per iteration.
        :type N: int
        :param num_iters: Number of optimization iterations, Defaults to 500.
        :type num_iters: int, optional
        :param lr: Adam optimizer learning rate, defaults to 1e-3.
        :type lr: float, optional
        """
        if num_units is None:
            num_units = max(2 * self.D, 15)

        q_theta = Architecture(
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

        # Initialize architecture to gaussian.
        q_theta.initialize(init_type, init_params)

        optimizer = tf.keras.optimizers.Adam(lr)

        eta = np.zeros((self.m,), np.float32)
        c = 10.

        @tf.function
        def train_step():
            with tf.GradientTape(persistent=True) as tape:
                x, log_q_x = q_theta(N)
                params = q_theta.trainable_variables
                tape.watch(params)
                T_x = self.eps(x)
                E_T_x = tf.reduce_mean(T_x, axis=0)
                R = E_T_x - mu
                E_log_q_x = tf.reduce_mean(log_q_x)
                cost_term1 = E_log_q_x + tf.reduce_sum(tf.multiply(eta, R))
                cost = cost_term1 + c / 2.0 * tf.reduce_sum(tf.square(R))

                R1 = tf.reduce_mean(T_x[: N // 2, :], 0) - mu
                R1s = tf.unstack(R1, axis=0)
                R2 = tf.reduce_mean(T_x[N // 2 :, :], 0) - mu

            grad1 = tape.gradient(cost_term1, params)
            gradR1s = tape.gradient(R1s[0], params)
            jacR1 = [[g] for g in gradR1s]
            for i in range(1, self.m):
                gradR1i = tape.gradient(R1s[i], params)
                for i, g in enumerate(gradR1i):
                    jacR1[i].append(g)

            jacR1 = [tf.stack(grad_list, axis=-1) for grad_list in jacR1]
            print(jacR1[0])
            grad2 = [tf.linalg.matvec(jacR1i, R2) for jacR1i in jacR1]

            gradients = [g1 + c * g2 for g1, g2 in zip(grad1, grad2)]
            optimizer.apply_gradients(zip(gradients, params))
            return cost

        for i in range(num_iters):
            loss = train_step()
            if i % 100 == 0:
                x, log_q_x = q_theta(N)
                print(i, "epi loss", loss)

        return q_theta

    def load_epi_dist(self,):
        raise NotImplementedError()

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
