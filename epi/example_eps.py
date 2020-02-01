"""Example emergent property statistics. """

import numpy as np
import tensorflow as tf


def linear2D_freq(a11, a12, a21, a22):
    """Linear 2D system frequency response characterisitcs.

    For a two-dimensional linear system:

    :math:`\\tau \\dot{x} = Ax`    
    
    :math:`A = \\begin{bmatrix} a_{11} & a_{12} \\\\ a_{21} & a_{22} \\end{bmatrix}`

    We can characterize the dynamics using the real and imaginary components of the
    primary eigenvalue :math:`\\lambda_1`, of C that which has the greates real component
    or secondarily the greatest imaginary component if the two eigenvalues are equal, 
    where :math:`C = A / \\tau`.

    :math:`T(x) = \\begin{bmatrix} \\text{real}(\\lambda_1) \\\\ \\text{real}(\\lambda_1 - E[\\text{real}(\\lambda_1)])^2 \\\\ 2\\pi\\text{imag}(\\lambda_1) \\\\ (2\\pi\\text{imag}(\\lambda_1) - \\mathbb{E}[2\\pi\\text{imag}(\\lambda_1)])^2 \\end{bmatrix}`

    :param a11: Dynamics coefficient.
    :type a11: tf.Tensor
    :param a12: Dynamics coefficient.
    :type a12: tf.Tensor
    :param a21: Dynamics coefficient.
    :type a21: tf.Tensor
    :param a22: Dynamics coefficient.
    :type a22: tf.Tensor
    """
    tau = 1.0
    c11 = a11 / tau
    c12 = a12 / tau
    c21 = a21 / tau
    c22 = a22 / tau

    real_term = 0.5 * (c11 + c22)
    complex_term = 0.5 * tf.sqrt(
        tf.complex(tf.square(c11 + c22) - 4.0 * (c11 * c22 - c12 * c21), 0.0)
    )
    alpha = real_term + tf.math.real(complex_term)
    omega = 2.0 * np.pi * tf.math.imag(complex_term)

    mean_alpha = tf.reduce_mean(alpha)
    mean_omega = tf.reduce_mean(omega)
    T_x = tf.stack(
        (alpha, tf.square(alpha - mean_alpha), omega, tf.square(omega - mean_omega)),
        axis=1,
    )

    return T_x
