""" Normalizing flow architecture class definitions for param distributions. """

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

from epi.error_formatters import format_type_err_msg, format_arch_type_err_msg
from epi.util import gaussian_backward_mapping


class Arch:
    def __init__(self, arch_type, num_layers, post_affine=True):
        self._set_arch_type(arch_type)
        self._set_num_layers(num_layers)
        self._set_post_affine(post_affine)
        self.trainable_variables = []

    def _set_arch_type(self, arch_type):  # Make this noninherited?
        arch_types = ["planar"]
        if type(arch_type) is not str:
            raise TypeError(format_type_err_msg(self, "arch_type", arch_type, str))
        if arch_type not in arch_types:
            raise ValueError(format_arch_type_err_msg(arch_type))
        self.arch_type = arch_type

    def _set_num_layers(self, num_layers):
        if type(num_layers) is not int:
            raise TypeError(format_type_err_msg(self, "num_layers", num_layers, int))
        elif num_layers < 1:
            raise ValueError(
                "Arch num_layers arg %d must be greater than 0." % num_layers
            )
        self.num_layers = num_layers

    def _set_post_affine(self, post_affine):
        if type(post_affine) is not bool:
            raise TypeError(format_type_err_msg(self, "post_affine", post_affine, bool))
        self.post_affine = post_affine

    def to_string(self,):
        arch_string = "%d_%s" % (self.num_layers, self.arch_type)
        if self.post_affine:
            arch_string += "_PA"
        return arch_string

    def to_model(self,):
        raise NotImplementedError()

    def initialize(self, init_type, init_params, N=200, num_iters=int(1e4), lr=1e-3):
        if (init_type == "iso_gauss"):
            loc = init_params['loc']
            scale = init_params['scale']
            eta = gaussian_backward_mapping(np.array(self.D*[loc]), scale*np.eye(self.D))
            print('eta', eta)
            optimizer = tf.keras.optimizers.Adam(lr)

            @tf.function
            def train_step():
                with tf.GradientTape() as tape:
                    x, log_q_x = self(N)
                    xl = x[:,:,tf.newaxis]
                    xr = x[:,tf.newaxis,:]
                    xxT = tf.matmul(xl, xr)
                    xxT_vec = tf.reshape(xxT, (N,self.D**2))
                    T_x = tf.concat((x, xxT_vec), axis=1)
                    E_T_x = tf.reduce_mean(T_x, axis=0)

                    H = tf.reduce_mean(-log_q_x)
                    loss = -H + - tf.reduce_sum(eta*E_T_x)

                params = self.trainable_variables
                gradients = tape.gradient(loss, params)

                optimizer.apply_gradients(zip(gradients, params))
                return loss

            for i in range(num_iters):
                loss = train_step()
                if (i % 1000 == 0):
                    print(i, loss)

        return None



class CouplingArch(Arch):
    def __init__(self, D, num_couplings, num_layers, num_units, post_affine=True):
        self._set_D(D)
        self._set_num_couplings(num_couplings)
        self._set_num_layers(num_layers)
        self._set_num_units(num_units)
        self._set_post_affine(post_affine)
        self.trainable_variables = []

        self.nvps = []
        self.permutations = []
        np.random.seed(0)
        for i in range(num_couplings):
            shift_and_log_scale_fn = tfb.real_nvp_default_template(
                hidden_layers=num_layers * [num_units]
            )

            nvp = tfb.RealNVP(
                num_masked=self.D // 2,
                shift_and_log_scale_fn=shift_and_log_scale_fn,
            )
            self.nvps.append(nvp)
            self.trainable_variables += nvp.trainable_variables

            if i < self.num_couplings - 1:
                self.permutations.append(tfb.Permute(np.random.permutation(self.D)))

        self.q0 = tfd.MultivariateNormalDiag(loc=self.D * [0.0])

        if self.post_affine:
            self.a = tf.Variable(initial_value=tf.ones((D,)), name='a')
            self.b = tf.Variable(initial_value=tf.zeros((D,)), name='b')

        self.trainable_variables = []

    @tf.function
    def __call__(self, N):

        x = self.q0.sample(N)
        log_q0 = self.q0.log_prob(x)

        sum_ldj = 0.0
        for i in range(self.num_couplings):
            nvp_i = self.nvps[i]
            sum_ldj += nvp_i.forward_log_det_jacobian(x, event_ndims=1)
            x = nvp_i(x)
            self.trainable_variables += nvp_i.trainable_variables
            log_q_x = log_q0 - sum_ldj
            if i < self.num_couplings - 1:
                x = self.permutations[i](x)

        if self.post_affine:
            self.scale = tfb.Scale(scale=self.a)
            self.shift = tfb.Shift(shift=self.b)
            self.PA = tfb.Chain([self.shift, self.scale])
            sum_ldj += self.PA.forward_log_det_jacobian(x, event_ndims=1)
            x = self.PA(x)
            self.trainable_variables += self.PA.trainable_variables

        log_q_x = log_q0 - sum_ldj
        return x, log_q_x

    def _set_D(self, D):
        if type(D) is not int:
            raise TypeError(format_type_err_msg(self, "D", D, int))
        elif D < 2:
            raise ValueError("CouplingArch D %d must be greater than 0." % D)
        self.D = D

    def _set_num_couplings(self, num_couplings):
        if type(num_couplings) is not int:
            raise TypeError(
                format_type_err_msg(self, "num_couplings", num_couplings, int)
            )
        elif num_couplings < 1:
            raise ValueError(
                "CouplingArch num_couplings %d must be greater than 0." % num_couplings
            )
        self.num_couplings = num_couplings

    def _set_num_units(self, num_units):
        if type(num_units) is not int:
            raise TypeError(format_type_err_msg(self, "num_units", num_units, int))
        elif num_units < 1:
            raise ValueError(
                "CouplingArch num_units %d must be greater than 0." % num_units
            )
        self.num_units = num_units

    def to_string(self,):
        arch_string = "NVP_%dC_%dL_%dU" % (
            self.num_couplings,
            self.num_layers,
            self.num_units,
        )
        if self.post_affine:
            arch_string += "_PA"
        return arch_string

class AutoregressiveArch(Arch):
    def __init__(self, D, num_ars, num_layers, num_units, post_affine=True):
        self._set_D(D)
        self._set_num_ars(num_ars)
        self._set_num_layers(num_layers)
        self._set_num_units(num_units)
        self._set_post_affine(post_affine)
        self.trainable_variables = []

        self.mafs = []
        self.permutations = []
        np.random.seed(0)
        for i in range(num_ars):
            shift_and_log_scale_fn = tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=num_layers * [num_units]
            )

            maf = tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=shift_and_log_scale_fn,
            )
            self.mafs.append(maf)
            self.trainable_variables += maf.trainable_variables

            if i < self.num_ars - 1:
                self.permutations.append(tfb.Permute(np.random.permutation(self.D)))

        self.q0 = tfd.MultivariateNormalDiag(loc=self.D * [0.0])

        if self.post_affine:
            self.a = tf.Variable(initial_value=tf.ones((D,)), name='a')
            self.b = tf.Variable(initial_value=tf.zeros((D,)), name='b')
            self.trainable_variables += [self.a, self.b]

    def __call__(self, N):

        x = self.q0.sample(N)
        log_q0 = self.q0.log_prob(x)

        sum_ldj = 0.0
        for i in range(self.num_ars):
            maf_i = self.mafs[i]
            sum_ldj += maf_i.forward_log_det_jacobian(x, event_ndims=1)
            x = maf_i(x)
            self.trainable_variables += maf_i.trainable_variables
            log_q_x = log_q0 - sum_ldj
            if i < self.num_ars - 1:
                x = self.permutations[i](x)

        if self.post_affine:
            self.scale = tfb.Scale(scale=self.a)
            self.shift = tfb.Shift(shift=self.b)
            self.PA = tfb.Chain([self.shift, self.scale])
            sum_ldj += self.PA.forward_log_det_jacobian(x, event_ndims=1)
            x = self.PA(x)
            self.trainable_variables += self.PA.trainable_variables



        log_q_x = log_q0 - sum_ldj
        return x, log_q_x

    def _set_D(self, D):
        if type(D) is not int:
            raise TypeError(format_type_err_msg(self, "D", D, int))
        elif D < 2:
            raise ValueError("AutoregressiveArch D %d must be greater than 0." % D)
        self.D = D

    def _set_num_ars(self, num_ars):
        if type(num_ars) is not int:
            raise TypeError(
                format_type_err_msg(self, "num_ars", num_ars, int)
            )
        elif num_ars < 1:
            raise ValueError(
                "AutoregressiveArch num_arss %d must be greater than 0." % num_ars
            )
        self.num_ars = num_ars

    def _set_num_units(self, num_units):
        if type(num_units) is not int:
            raise TypeError(format_type_err_msg(self, "num_units", num_units, int))
        elif num_units < 1:
            raise ValueError(
                "AutoregressiveArch num_units %d must be greater than 0." % num_units
            )
        self.num_units = num_units

    def to_string(self,):
        arch_string = "MAF_%dAR_%dL_%dU" % (
            self.num_ars,
            self.num_layers,
            self.num_units,
        )
        if self.post_affine:
            arch_string += "_PA"
        return arch_string

