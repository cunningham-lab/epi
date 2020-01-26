""" Normalizing flow architecture class definitions for param distributions. """

import numpy as np
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

from epi.error_formatters import format_type_err_msg
from epi.util import gaussian_backward_mapping


class Architecture(tf.keras.Model):
    def __init__(
        self,
        arch_type,
        D,
        num_stages,
        num_layers,
        num_units,
        batch_norm=True,
        post_affine=True,
    ):
        super().__init__()
        self._set_arch_type(arch_type)
        self._set_D(D)
        self._set_num_stages(num_stages)
        self._set_num_layers(num_layers)
        self._set_num_units(num_units)
        self._set_batch_norm(batch_norm)
        self._set_post_affine(post_affine)
        self.bn_momentum = 0.0

        self.stages = []
        self.permutations = []
        if self.batch_norm:
            self.batch_norms = []

        np.random.seed(0)
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

        self.q0 = tfd.MultivariateNormalDiag(loc=self.D * [0.0])

        if self.post_affine:
            self.a = tf.Variable(initial_value=tf.ones((D,)), name="a")
            self.b = tf.Variable(initial_value=tf.zeros((D,)), name="b")

    @tf.function
    def call(self, x, log_q0):

        #x = self.q0.sample(N)
        #log_q0 = self.q0.log_prob(x)

        sum_ldj = 0.0
        for i in range(self.num_stages):
            stage_i = self.stages[i]
            sum_ldj += stage_i.forward_log_det_jacobian(x, event_ndims=1)
            x = stage_i(x)
            print(i, stage_i.trainable_variables)
            for var in stage_i.trainable_variables:
                print(var)
                self.trainable_variables.append(var)
                print(self.trainable_variables)
            if i < self.num_stages - 1:
                if self.batch_norm:
                    batch_norm_i = self.batch_norms[i]
                    sum_ldj += batch_norm_i.forward_log_det_jacobian(x, event_ndims=1)
                    x = batch_norm_i(x)
                x = self.permutations[i](x)

        if self.post_affine:
            self.scale = tfb.Scale(scale=self.a)
            self.shift = tfb.Shift(shift=self.b)
            self.PA = tfb.Chain([self.shift, self.scale])
            sum_ldj += self.PA.forward_log_det_jacobian(x, event_ndims=1)
            x = self.PA(x)
            #self.trainable_variables += self.PA.trainable_variables

        log_q_x = log_q0 - sum_ldj
        return x, log_q_x

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

    def _set_post_affine(self, post_affine):
        if type(post_affine) is not bool:
            raise TypeError(format_type_err_msg(self, "post_affine", post_affine, bool))
        self.post_affine = post_affine

    def to_string(self,):
        arch_string = "%d_%s" % (self.num_layers, self.arch_type)
        if self.post_affine:
            arch_string += "_PA"
        return arch_string

    def initialize(
        self,
        init_type,
        init_params,
        N=500,
        num_iters=int(1e4),
        lr=1e-3,
        KL_th=None,
        verbose=False,
    ):

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
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=self)
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=None)

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
            print('params', params)
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
                if KL < KL_th:
                    print("Finished initializing")
                    return manager.save()

                    

        print("Final KL to target after initialization optimization: %.2E." % KL)
        return manager.save()

    def to_string(self,):
        if self.arch_type == "coupling":
            arch_type_str = "C"
        elif self.arch_type == "autoregressive":
            arch_type_str = "AR"
        arch_string = "%d%s_%dL_%dU" % (
            self.num_stages,
            arch_type_str,
            self.num_layers,
            self.num_units,
        )
        if self.post_affine:
            arch_string += "_PA"
        return arch_string
