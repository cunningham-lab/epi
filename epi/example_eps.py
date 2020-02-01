"""Example emergent property statistics. """

import numpy as np
import tensorflow as tf

def linear2D_freq(a11, a12, a21, a22):
    """Linear 2D system frequency response characterisitcs.

    :math:`E_{x\\sim p(x \\mid z)}\\left[T(x)\\right] = f_{p,T}(z) = E \\begin{bmatrix} \\text{real}(\\lambda_1) \\\\\\\\
    \\frac{\\text{imag}(\\lambda_1)}{2\pi} \\\\\\\\ \\text{real}(\\lambda_1)^2 \\\\\\\\
    (\\frac{\\text{imag}(\\lambda_1)}{2\pi}^2 \end{bmatrix}`

    :param a11: Dynamics coefficient.
    :type a11: tf.Tensor
    :param a12: Dynamics coefficient.
    :type a12: tf.Tensor
    :param a21: Dynamics coefficient.
    :type a21: tf.Tensor
    :param a22: Dynamics coefficient.
    :type a22: tf.Tensor
    """
    tau = 1.
    c11 = a11 / tau
    c12 = a12 / tau
    c21 = a21 / tau
    c22 = a22 / tau

    real_term = 0.5 * (c11 + c22)
    complex_term = 0.5*tf.sqrt(tf.complex(tf.square(c11 + c22) - 4.*(c11*c22 - c12*c21), 0.))
    alpha = real_term + tf.math.real(complex_term)
    omega = 2.*np.pi*tf.math.imag(complex_term)

    mean_alpha = tf.reduce_mean(alpha)
    mean_omega = tf.reduce_mean(omega)
    T_x = tf.stack((alpha, tf.square(alpha-mean_alpha), omega, tf.square(omega-mean_omega)), axis=1)

    return T_x
