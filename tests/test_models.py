""" Test models. """

import numpy as np
import tensorflow as tf
import scipy.stats
from epi.models import Parameter, Model, Distribution
from epi.example_eps import linear2D_freq
from epi.normalizing_flows import NormalizingFlow
from pytest import raises


def test_Parameter_init():
    """Test Parameter initialization."""
    b1 = (-0.1, 1.2)
    p = Parameter("foo", b1)
    assert p.name == "foo"
    assert type(p.bounds) is tuple
    assert p.bounds[0] == -0.1
    assert p.bounds[1] == 1.2
    assert len(p.bounds) == 2

    with raises(TypeError):
        p = Parameter(20, b1)

    p = Parameter("bar", [-0.1, 1.2])
    assert p.bounds == (-0.1, 1.2)
    p = Parameter("bar", np.array([-0.1, 1.2]))
    assert p.bounds == (-0.1, 1.2)

    with raises(TypeError):
        p = Parameter("foo", "bar")

    with raises(ValueError):
        p = Parameter("foo", [1, 2, 3])

    with raises(TypeError):
        p = Parameter("foo", ["a", "b"])
    with raises(TypeError):
        p = Parameter("foo", [1, "b"])

    with raises(ValueError):
        p = Parameter("foo", [1, -1])

    with raises(ValueError):
        p = Parameter("foo", [1, 1])

    return None


def test_Model_init():
    """Test Model initialization."""
    p1 = Parameter("a", [0, 1])
    p2 = Parameter("b")
    params = [p1, p2]
    M = Model("foo", params)
    assert M.name == "foo"
    for i, p in enumerate(M.parameters):
        assert p == params[i]

    with raises(TypeError):
        Model(1, params)

    params = [p1, "bar"]
    with raises(TypeError):
        Model("foo", params)

    p3 = Parameter("c", [1, 4])
    p3.bounds = (1, -1)
    params = [p1, p2, p3]
    with raises(ValueError):
        Model("foo", params)

    p3.bounds = (1, 1)
    with raises(ValueError):
        Model("foo", params)

    p3 = Parameter("a", [1, 4])
    params = [p1, p2, p3]
    with raises(ValueError):
        Model("foo", params)

    return None


def test_epi():
    mu = np.array([0.0, 0.1, 2 * np.pi, 0.1 * np.pi])

    lb_a12 = 0.0
    ub_a12 = 10.0
    lb_a21 = -10.0
    ub_a21 = 0.0
    a11 = Parameter("a11", [0.0, np.PINF])
    a12 = Parameter("a12", [lb_a12, ub_a12])
    a21 = Parameter("a21", [lb_a21, ub_a21])
    a22 = Parameter("a22", [np.NINF, 0.0])
    params = [a11, a12, a21, a22]

    M = Model("lds", params)
    m = 4
    M.set_eps(linear2D_freq, m)
    q_theta, opt_data = M.epi(mu, num_iters=100)

    z = q_theta(1000)
    log_q_z = q_theta.log_prob(z)
    assert np.sum(z[:, 0] < 0.0) == 0
    assert np.sum(z[:, 1] < lb_a12) == 0
    assert np.sum(z[:, 1] > ub_a12) == 0
    assert np.sum(z[:, 2] < lb_a21) == 0
    assert np.sum(z[:, 2] > ub_a21) == 0
    assert np.sum(z[:, 3] > 0.0) == 0
    assert np.sum(1 - np.isfinite(z)) == 0
    assert np.sum(1 - np.isfinite(log_q_z)) == 0

    opt_data_cols = ["k", "iteration", "H"] + ["R%d" % i for i in range(1, m + 1)]
    for x, y in zip(opt_data.columns, opt_data_cols):
        assert x == y

    # Intentionally swap order in list to insure proper handling.
    params = [a22, a21, a12, a11]
    M = Model("lds2", params)
    M.set_eps(linear2D_freq, m)
    q_theta, opt_data = M.epi(mu, num_iters=100)

    z = q_theta(1000)
    log_q_z = q_theta.log_prob(z)
    assert np.sum(z[:, 0] < 0.0) == 0
    assert np.sum(z[:, 1] < lb_a12) == 0
    assert np.sum(z[:, 1] > ub_a12) == 0
    assert np.sum(z[:, 2] < lb_a21) == 0
    assert np.sum(z[:, 2] > ub_a21) == 0
    assert np.sum(z[:, 3] > 0.0) == 0
    assert np.sum(1 - np.isfinite(z)) == 0
    assert np.sum(1 - np.isfinite(log_q_z)) == 0

    for x, y in zip(opt_data.columns, opt_data_cols):
        assert x == y

    return None


def test_Distribution():
    """ Test Distribution class."""
    Ds = [2, 4]
    num_dists = 3
    N1 = 1000
    N2 = 10
    for D in Ds:
        df = 2 * D
        inv_wishart = scipy.stats.invwishart(df=df, scale=df * np.eye(D))
        for i in range(num_dists):
            nf = NormalizingFlow(
                "autoregressive",
                D,
                1,
                2,
                max(10, D),
                batch_norm=False,
                post_affine=True,
            )
            mu = np.random.normal(0.0, 1.0, (D, 1))
            Sigma = inv_wishart.rvs(1)
            mvn = scipy.stats.multivariate_normal(mu[:, 0], Sigma)
            init_type = "gaussian"
            init_params = {"mu": mu, "Sigma": Sigma}
            opt_df = nf.initialize(
                init_type, init_params, num_iters=5000, load_if_cached=False, save=False
            )
            q_theta = Distribution(nf)

            z = q_theta.sample(N1)
            assert np.isclose(np.mean(z, axis=0), mu[:, 0], rtol=0.1).all()
            cov = np.cov(z.T)
            assert np.sum(np.square(cov - Sigma)) / np.sum(np.square(Sigma)) < 0.1

            z = q_theta.sample(N2)
            assert np.isclose(mvn.logpdf(z), q_theta.log_prob(z), rtol=0.1).all()

            Sigma_inv = np.linalg.inv(Sigma)

            # Test gradient
            grad_true = np.dot(Sigma_inv, mu - z.T).T
            grad_z = q_theta.gradient(z)
            assert (
                np.sum(np.square(grad_true - grad_z)) / np.sum(np.square(grad_true))
                < 0.1
            )

            # Test hessian
            hess_true = np.array(N2 * [-Sigma_inv])
            hess_z = q_theta.hessian(z)
            assert (
                np.sum(np.square(hess_true - hess_z)) / np.sum(np.square(hess_true))
                < 0.1
            )
