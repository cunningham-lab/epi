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
            zs = tf.unstack(z[:, tf.newaxis, :], axis=2)
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
        post_affine=True,
        random_seed=1,
        init_type="iso_gauss",
        init_params={"loc": 0.0, "scale": 1.0},
        N=500,
        lr=1e-3,
        num_iters=10000,
    ):

        if num_units is None:
            num_units = max(2 * self.D, 15)

        q_theta = Architecture(
            arch_type=arch_type,
            D=self.D,
            num_stages=num_stages,
            num_layers=num_layers,
            num_units=num_units,
            batch_norm=batch_norm,
            post_affine=post_affine,
            bounds=self._get_bounds(),
            random_seed=random_seed,
        )

        # Initialize architecture to gaussian.
        q_theta.initialize(init_type, init_params)

        optimizer = tf.keras.optimizers.Adam(lr)

        eta = np.zeros((self.m,), np.float32)
        c = 1e-3

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

        return q_theta, train_step

    def load_epi_dist(self,):
        raise NotImplementedError()

    def parameter_check(self, parameters, verbose=False):
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
