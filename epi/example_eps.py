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
    real_lambda = real_term + tf.math.real(complex_term)
    imag_lambda = tf.math.imag(complex_term)

    T_x = tf.concat(
        (
            real_lambda,
            tf.square(real_lambda - 0.0),
            imag_lambda,
            tf.square(imag_lambda - (2.0 * np.pi)),
        ),
        axis=1,
    )
    return T_x


def linear2D_freq_sq(A):
    tau = 1.0
    c11 = A[:, 0] / tau
    c12 = A[:, 1] / tau
    c21 = A[:, 2] / tau
    c22 = A[:, 3] / tau

    real_term = 0.5 * (c11 + c22)
    complex_term = 0.5 * tf.sqrt(
        tf.complex(tf.square(c11 + c22) - 4.0 * (c11 * c22 - c12 * c21), 0.0)
    )
    real_lambda = real_term + tf.math.real(complex_term)
    imag_lambda = tf.math.imag(complex_term)

    T_x = tf.stack(
        (
            real_lambda,
            tf.square(real_lambda - 0.0),
            imag_lambda,
            tf.square(imag_lambda - (2.0 * np.pi)),
        ),
        axis=1,
    )
    return T_x


def linear2D_freq_np(a11, a12, a21, a22):
    tau = 1.0
    C = np.array([[a11, a12], [a21, a22]]) / tau
    eigs = np.linalg.eigvals(C)
    eig1, eig2 = eigs[0], eigs[1]
    eig1_r = np.real(eig1)
    eig1_i = np.imag(eig1)
    eig2_r = np.real(eig2)
    eig2_i = np.imag(eig2)
    if eig1_r >= eig2_r:
        real_lambda = eig1_r
    else:
        real_lambda = eig2_r

    if eig1_i >= eig2_i:
        imag_lambda = eig1_i

    return real_lambda, imag_lambda


X_INIT = tf.constant(np.random.normal(1.0, 0.01, (1, 4, 1)).astype(np.float32))

def V1_dr_eps(alpha, inc_val, b=np.array([1., 1., 1., 1.25])):
    neuron_inds = {"E": 0, "P": 1, "S": 2, "V": 3}
    neuron_ind = neuron_inds[alpha]
    b = tf.constant(b[None,:,None], dtype=tf.float32)

    def dr(dh):
        dh = dh[:, :, None]

        dt = 0.005
        T = 100
        tau = 0.02

        h = b + dh

        _x_shape = tf.ones_like(dh, dtype=tf.float32)
        x_init = _x_shape*X_INIT

        npzfile = np.load("data/V1_Zs.npz")
        _W = npzfile["Z_allen_square"][None, :, :]
        _W[:, :, 1:] = -_W[:, :, 1:]
        W = tf.constant(_W, dtype=tf.float32)

        def f1(y):
            return (-y + (tf.nn.relu(tf.matmul(W, y) + b) ** 2.0)) / tau

        def f2(y):
            return (-y + (tf.nn.relu(tf.matmul(W, y) + h) ** 2.0)) / tau

        x1 = euler_sim(f1, x_init, dt, T)[:, neuron_ind]
        x2 = euler_sim(f2, x_init, dt, T)[:, neuron_ind]

        diff = x2 - x1
        T_x = tf.stack((diff, (diff - inc_val) ** 2), axis=1)

        return T_x

    return dr

def V1_all_dr_eps(inc_val, b=np.array([1., 1., 1., 1.25])):

    b = tf.constant(b[None,:,None], dtype=tf.float32)

    def dr(dh):
        dh = dh[:, :, None]

        dt = 0.005
        T = 100
        tau = 0.02

        h = b + dh

        _x_shape = tf.ones_like(dh, dtype=tf.float32)
        x_init = _x_shape*X_INIT

        npzfile = np.load("data/V1_Zs.npz")
        _W = npzfile["Z_allen_square"][None, :, :]
        _W[:, :, 1:] = -_W[:, :, 1:]
        W = tf.constant(_W, dtype=tf.float32)

        def f1(y):
            return (-y + (tf.nn.relu(tf.matmul(W, y) + b) ** 2.0)) / tau

        def f2(y):
            return (-y + (tf.nn.relu(tf.matmul(W, y) + h) ** 2.0)) / tau

        x1 = euler_sim(f1, x_init, dt, T)
        x2 = euler_sim(f2, x_init, dt, T)

        diff = x2 - x1
        T_x = tf.concat((diff, (diff - inc_val) ** 2), axis=1)

        return T_x
    return dr


def V1_ISN(dh, b=np.array([1., 1., 1., 1.25])):

    dh = dh[:, :, None]
    b = tf.constant(b[None,:,None], dtype=tf.float32)

    dt = 0.005
    T = 100
    tau = 0.02

    h = b + dh

    _x_shape = tf.ones_like(dh, dtype=tf.float32)
    x_init = _x_shape*X_INIT

    npzfile = np.load("data/V1_Zs.npz")
    _W = npzfile["Z_allen_square"][None, :, :]
    _W[:, :, 1:] = -_W[:, :, 1:]
    W = tf.constant(_W, dtype=tf.float32)

    def f(y):
        return (-y + (tf.nn.relu(tf.matmul(W, y) + h) ** 2.0)) / tau

    r_ss_E = euler_sim(f, x_init, dt, T)[:,0]
    u_E = tf.math.sqrt(r_ss_E)
    ISN = (1. - 2. * u_E * _W[0, 0, 0])[:,None]
    return ISN



def V1_sim(dh, b=np.array([1., 1., 1., 1.25])):

    dh = dh[:, :, None]
    b = tf.constant(b[None,:,None], dtype=tf.float32)

    dt = 0.005
    T = 100
    tau = 0.02

    h = b + dh

    _x_shape = tf.ones_like(dh, dtype=tf.float32)
    x_init = _x_shape*X_INIT

    npzfile = np.load("data/V1_Zs.npz")
    _W = npzfile["Z_allen_square"][None, :, :]
    _W[:, :, 1:] = -_W[:, :, 1:]
    W = tf.constant(_W, dtype=tf.float32)

    def f(y):
        return (-y + (tf.nn.relu(tf.matmul(W, y) + h) ** 2.0)) / tau

    r_t = euler_sim_traj(f, x_init, dt, T)
    return r_t

def euler_sim(f, x_init, dt, T):
    x = x_init
    for t in range(T):
        x = x + f(x) * dt
    return x[:, :, 0]

def euler_sim_traj(f, x_init, dt, T):
    x = x_init
    xs = [x_init]
    for t in range(T):
        x = x + f(x) * dt
        xs.append(x)
    return tf.concat(xs, axis=2)
