""" Test util functions. """

import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.normalizing_flows import NormalizingFlow
from epi.util import (
    gaussian_backward_mapping,
    np_column_vec,
    array_str,
    aug_lag_vars,
    unbiased_aug_grad,
    AugLagHPs,
    sample_aug_lag_hps,
    get_hash,
    set_dir_index,
    get_dir_index,
)
from epi.example_eps import linear2D_freq, linear2D_freq_np
import pytest
from pytest import raises
import os

DTYPE = np.float32


def test_get_hash():
    x = np.random.normal(0., 1., (10,))
    y = np.random.normal(0., 1., (10,))
    z = 'foo'

    h1 = get_hash([x])
    h2 = get_hash([y])
    h3 = get_hash([x+(1e-16)])
    h4 = get_hash([x, y])
    h5 = get_hash([x, y])
    h6 = get_hash([x, y, z])
    h7 = get_hash([x, y, z, None])
    h8 = get_hash([None, x, y, z])

    assert(h1 != h2)
    assert(h1 != h3)
    assert(h1 != h4)
    assert(h4 == h5)
    assert(h4 != h6)
    assert(h6 == h7)
    assert(h6 == h8)
    return None

def test_dir_indexes():
    np.random.seed(0)
    index = {
        "w": None,
        "x": np.random.normal(0., 1., (10,)),
        "y": np.random.normal(0., 1., (10,)),
        "z": 'foo',
    }
    index_file = os.path.join("foo.pkl")
    set_dir_index(index, index_file)
    set_dir_index(index, index_file)
    _index = get_dir_index(index_file)
    for key, value in index.items():
        if type(value) is np.ndarray:
            assert(np.isclose(_index[key], value).all())
        else:
            assert(_index[key] == value)

    index_file = "temp1.foo.pkl"
    assert(get_dir_index(index_file) is None)
    return None
    

def test_gaussian_backward_mapping():
    """ Test gaussian_backward_mapping. """
    eta = gaussian_backward_mapping(np.zeros(2), np.eye(2))
    eta_true = np.array([0.0, 0.0, -0.5, 0.0, 0.0, -0.5])
    assert np.equal(eta, eta_true).all()

    eta = gaussian_backward_mapping(np.zeros(2), 2.0 * np.eye(2))
    eta_true = np.array([0.0, 0.0, -0.25, 0.0, 0.0, -0.25])
    assert np.equal(eta, eta_true).all()

    eta = gaussian_backward_mapping(2.0 * np.ones(2), np.eye(2))
    eta_true = np.array([2.0, 2.0, -0.5, 0.0, 0.0, -0.5])
    assert np.equal(eta, eta_true).all()

    mu = np.array([-4.0, 1000, 0.0001])
    Sigma = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    eta_true = np.concatenate(
        (np.dot(Sigma_inv, mu), np.reshape(-0.5 * Sigma_inv, (9,)))
    )
    eta = gaussian_backward_mapping(mu, Sigma)
    assert np.equal(eta, eta_true).all()

    with raises(TypeError):
        gaussian_backward_mapping("foo", np.eye(2))
    with raises(TypeError):
        gaussian_backward_mapping(np.zeros(2), "foo")

    with raises(ValueError):
        gaussian_backward_mapping(np.zeros(2), np.random.normal(0.0, 1.0, (2, 2, 2)))
    with raises(ValueError):
        gaussian_backward_mapping(np.zeros(2), np.random.normal(0.0, 1.0, (2, 3)))
    with raises(ValueError):
        gaussian_backward_mapping(np.zeros(2), np.random.normal(0.0, 1.0, (2, 2)))
    with raises(ValueError):
        eta = gaussian_backward_mapping(np.zeros(2), np.eye(3))
    assert np.equal(eta, eta_true).all()

    return None


def test_np_column_vec():
    """ Test np_column_vec. """
    x = np.random.uniform(-100, 100, (20, 1))
    X = np.random.uniform(-100, 100, (20, 20))
    X3 = np.random.uniform(-100, 100, (10, 10, 10))
    assert np.equal(x, np_column_vec(x[:, 0])).all()
    assert np.equal(x, np_column_vec(x.T)).all()

    with raises(TypeError):
        np_column_vec("foo")

    with raises(ValueError):
        np_column_vec(X)

    with raises(ValueError):
        np_column_vec(X3)
    return None


def test_array_str():
    a = np.array([0.0492])
    a_str = "4.92E-02"
    assert a_str == array_str(a)

    a = np.array([3.2, 15.2, -0.4, -9.2])
    a_str = "3.20E+00_1.52E+01_-4.00E-01_-9.20E+00"
    assert a_str == array_str(a)

    a = np.array([3.2, 3.2, 15.2, -0.4, -9.2])
    a_str = "2x3.20E+00_1.52E+01_-4.00E-01_-9.20E+00"
    assert a_str == array_str(a)

    a = np.array([3.2, 15.2, 15.2, 15.2, -0.4, -9.2])
    a_str = "3.20E+00_3x1.52E+01_-4.00E-01_-9.20E+00"
    assert a_str == array_str(a)

    a = np.array([3.2, 15.2, -0.4, -9.2, -9.2, -9.2, -9.2])
    a_str = "3.20E+00_1.52E+01_-4.00E-01_4x-9.20E+00"
    assert a_str == array_str(a)

    a = np.array([3.2, 3.2, 15.2, -0.4, -9.2, -9.2])
    a_str = "2x3.20E+00_1.52E+01_-4.00E-01_2x-9.20E+00"
    assert a_str == array_str(a)

    with raises(TypeError):
        array_str("foo")

    with raises(ValueError):
        array_str(np.random.normal(0.0, 1.0, (3, 3)))

    return None

def test_aug_lag_vars():
    # Test using linear 2D system eps
    N = 100
    z = np.random.normal(0.0, 1.0, (N, 4)).astype(DTYPE)
    log_q_z = np.random.normal(2.0, 3.0, (N,)).astype(DTYPE)
    mu = np.array([0.0, 0.1, 2 * np.pi, 0.1 * np.pi]).astype(DTYPE)

    lb = np.NINF
    ub = np.PINF
    a11 = Parameter("a11", 1, lb, ub)
    a12 = Parameter("a12", 1, lb, ub)
    a21 = Parameter("a21", 1, lb, ub)
    a22 = Parameter("a22", 1, lb, ub)
    params = [a11, a12, a21, a22]
    M = Model("lds", params)
    M.set_eps(linear2D_freq)

    H, R, R1s, R2 = aug_lag_vars(z, log_q_z, M.eps, mu, N)

    alphas = np.zeros((N,))
    omegas = np.zeros((N,))
    for i in range(N):
        alphas[i], omegas[i] = linear2D_freq_np(z[i, 0], z[i, 1], z[i, 2], z[i, 3])

    # mean_alphas = np.mean(alphas)
    # mean_omegas = np.mean(omegas)
    mean_alphas = 0.0
    mean_omegas = 2.0 * np.pi

    T_x_np = np.stack(
        (
            alphas,
            np.square(alphas - mean_alphas),
            omegas,
            np.square(omegas - mean_omegas),
        ),
        axis=1,
    )

    H_np = np.mean(-log_q_z)
    R_np = np.mean(T_x_np, 0) - mu
    R1_np = np.mean(T_x_np[: N // 2, :], 0) - mu
    R2_np = np.mean(T_x_np[N // 2 :, :], 0) - mu
    R1s_np = list(R1_np)

    rtol = 1e-3
    assert np.isclose(H, H_np, rtol=rtol)
    assert np.isclose(R, R_np, rtol=rtol).all()
    assert np.isclose(R1s, R1s_np, rtol=rtol).all()
    assert np.isclose(R2, R2_np, rtol=rtol).all()

    return None


def test_unbiased_aug_grad():
    # Test using linear 2D system eps
    N = 100
    z = np.random.normal(0.0, 1.0, (N, 4)).astype(DTYPE)
    log_q_z = np.random.normal(2.0, 3.0, (N,)).astype(DTYPE)
    mu = np.array([0.0, 0.1, 2 * np.pi, 0.1 * np.pi]).astype(DTYPE)

    lb = np.NINF
    ub = np.PINF
    a11 = Parameter("a11", 1, lb, ub)
    a12 = Parameter("a12", 1, lb, ub)
    a21 = Parameter("a21", 1, lb, ub)
    a22 = Parameter("a22", 1, lb, ub)
    params = [a11, a12, a21, a22]
    M = Model("lds", params)
    M.set_eps(linear2D_freq)

    nf = NormalizingFlow(
        arch_type="autoregressive", D=4, num_stages=1, num_layers=2, num_units=15
    )

    with tf.GradientTape(persistent=True) as tape:
        z, log_q_z = nf(N)
        params = nf.trainable_variables
        nparams = len(params)
        tape.watch(params)
        _, _, R1s, R2 = aug_lag_vars(z, log_q_z, M.eps, mu, N)
        aug_grad = unbiased_aug_grad(R1s, R2, params, tape)

        T_x_grads = [
            [[None for i in range(N // 2)] for i in range(4)] for i in range(nparams)
        ]
        T_x = M.eps(z)
        for i in range(N // 2):
            T_x_i_grads = []
            for j in range(4):
                _grads = tape.gradient(T_x[i, j] - mu[j], params)
                for k in range(nparams):
                    T_x_grads[k][j][i] = _grads[k]
    del tape

    # Average across the first half of samples
    for k in range(nparams):
        T_x_grads[k] = np.mean(np.array(T_x_grads[k]), axis=1)

    R2_np = np.mean(T_x[N // 2 :, :], 0) - mu
    aug_grad_np = []
    for k in range(nparams):
        aug_grad_np.append(np.tensordot(T_x_grads[k], R2_np, axes=(0, 0)))

    for i in range(nparams):
        assert np.isclose(aug_grad_np[i], aug_grad[i], rtol=1e-3).all()

    return None


def test_AugLagHPs():
    with raises(TypeError):
        AugLagHPs(N="foo")
    with raises(ValueError):
        AugLagHPs(N=0)

    with raises(TypeError):
        AugLagHPs(lr="foo")
    with raises(ValueError):
        AugLagHPs(lr=-1.0)

    with raises(TypeError):
        AugLagHPs(c0="foo")
    with raises(ValueError):
        AugLagHPs(c0=-1.0)

    with raises(TypeError):
        AugLagHPs(gamma="foo")
    with raises(ValueError):
        AugLagHPs(gamma=-0.1)

    with raises(TypeError):
        AugLagHPs(beta="foo")
    with raises(ValueError):
        AugLagHPs(beta=-1.0)


def test_sample_aug_lag_hps():
    N_bounds = [100, 500]
    lr_bounds = [1e-3, 0.5]
    c0_bounds = [1e-10, 1e-6]
    gamma_bounds = [0.25, 0.4]

    n = 100
    aug_lag_hps = sample_aug_lag_hps(n)

    aug_lag_hps = sample_aug_lag_hps(n, N_bounds, lr_bounds, c0_bounds, gamma_bounds)
    for i in range(n):
        aug_lag_hp = aug_lag_hps[i]
        assert aug_lag_hp.N >= N_bounds[0]
        assert aug_lag_hp.N < N_bounds[1]
        assert aug_lag_hp.lr >= lr_bounds[0]
        assert aug_lag_hp.lr < lr_bounds[1]
        assert aug_lag_hp.c0 >= c0_bounds[0]
        assert aug_lag_hp.c0 < c0_bounds[1]
        assert aug_lag_hp.gamma >= gamma_bounds[0]
        assert aug_lag_hp.gamma < gamma_bounds[1]
        assert aug_lag_hp.beta == 1.0 / aug_lag_hp.gamma

    aug_lag_hp = sample_aug_lag_hps(1, N_bounds, lr_bounds, c0_bounds, gamma_bounds)
    assert aug_lag_hp.N >= N_bounds[0]
    assert aug_lag_hp.N < N_bounds[1]
    assert aug_lag_hp.lr >= lr_bounds[0]
    assert aug_lag_hp.lr < lr_bounds[1]
    assert aug_lag_hp.c0 >= c0_bounds[0]
    assert aug_lag_hp.c0 < c0_bounds[1]
    assert aug_lag_hp.gamma >= gamma_bounds[0]
    assert aug_lag_hp.gamma < gamma_bounds[1]
    assert aug_lag_hp.beta == 1.0 / aug_lag_hp.gamma

    with raises(TypeError):
        sample_aug_lag_hps(n, N_bounds="foo")

    with raises(ValueError):
        sample_aug_lag_hps(n, N_bounds=[1, 2, 3])

    with raises(ValueError):
        sample_aug_lag_hps(n, N_bounds=[2, 1])

    with raises(ValueError):
        sample_aug_lag_hps(n, N_bounds=[2, 100])

    with raises(ValueError):
        sample_aug_lag_hps(n, N_bounds=[2, 100])

    with raises(ValueError):
        sample_aug_lag_hps(n, lr_bounds=[-0.1, 0.1])

    with raises(ValueError):
        sample_aug_lag_hps(n, c0_bounds=[-0.1, 0.1])

    with raises(ValueError):
        sample_aug_lag_hps(n, gamma_bounds=[-0.1, 0.1])

    return None

