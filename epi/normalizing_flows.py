""" Normalizing flow architecture class definitions for param distributions. """

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

from epi.error_formatters import format_type_err_msg, format_arch_type_err_msg


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
            self.shift_and_log_scale_fn = tfb.real_nvp_default_template(
                hidden_layers=num_layers * [num_units]
            )

            nvp = tfb.RealNVP(
                num_masked=self.D // 2,
                shift_and_log_scale_fn=self.shift_and_log_scale_fn,
            )
            self.nvps.append(nvp)
            self.trainable_variables += nvp.trainable_variables

            if i < self.num_couplings - 1:
                self.permutations.append(tfb.Permute(np.random.permutation(self.D)))

        self.q0 = tfd.MultivariateNormalDiag(loc=self.D * [0.0])

        if self.post_affine:
            self.a = tf.Variable(tf.ones((D,)))
            self.b = tf.Variable(tf.zeros((D,)))
            self.trainable_variables += [self.a, self.b]

    def __call__(self, N):

        x = self.q0.sample(N)
        log_q0 = self.q0.log_prob(x)

        sum_ldj = 0.0
        for i in range(self.num_couplings):
            sum_ldj += self.nvps[i].forward_log_det_jacobian(x, event_ndims=1)
            x = self.nvps[i](x)
            log_q_x = log_q0 - sum_ldj
            if i < self.num_couplings - 1:
                x = self.permutations[i](x)

        if self.post_affine:
            self.scale = tfb.Scale(scale=self.a)
            self.shift = tfb.Shift(shift=self.b)
            self.PA = tfb.Chain([self.shift, self.scale])

            sum_ldj += self.PA.forward_log_det_jacobian(x, event_ndims=1)
            x = self.PA(x)

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
