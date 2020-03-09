""" Test models. """

import numpy as np
import tensorflow as tf
import scipy.stats
import pandas as pd
import os
from epi.models import Parameter, Model, Distribution
from epi.normalizing_flows import NormalizingFlow
from epi.util import AugLagHPs
from epi.example_eps import linear2D_freq
from pytest import raises


def test_Parameter_init():
    """Test Parameter initialization."""
    p = Parameter("foo", 1, -0.1, 1.2)
    assert p.name == "foo"
    assert p.D == 1
    assert p.lb[0] == -0.1
    assert p.ub[0] == 1.2

    p = Parameter("foo", 4, -np.random.rand(4), np.random.rand(4) + 1)
    assert p.name == "foo"
    assert p.D == 4

    with raises(TypeError):
        p = Parameter(20, 1)
    with raises(TypeError):
        p = Parameter("foo", "bar")
    with raises(TypeError):
        p = Parameter("foo", 1, "bar")
    with raises(TypeError):
        p = Parameter("foo", 1, 0.0, "bar")

    with raises(ValueError):
        p = Parameter("foo", -1)
    with raises(ValueError):
        p = Parameter("foo", 1, 1.0, 0.0)
    with raises(ValueError):
        p = Parameter("foo", 1, 0.0, 0.0)
    with raises(ValueError):
        p = Parameter("foo", 2, lb=np.random.rand(3))
    with raises(ValueError):
        p = Parameter("foo", 2, lb=np.random.rand(2, 2))
    with raises(ValueError):
        p = Parameter("foo", 2, ub=np.random.rand(3))
    with raises(ValueError):
        p = Parameter("foo", 2, ub=np.random.rand(2, 2))

    return None


def test_Model_init():
    """Test Model initialization."""
    p1 = Parameter("a", 1, 0, 1)
    p2 = Parameter("b", 1)
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

    with raises(TypeError):
        Model("foo", "bar")

    p3 = Parameter("c", 1, 1, 4)
    p3.lb = np.array([1])
    p3.ub = np.array([-1])
    params = [p1, p2, p3]
    with raises(ValueError):
        Model("foo", params)

    p3.lb = np.array([1])
    p3.ub = np.array([1])
    with raises(ValueError):
        Model("foo", params)

    p3 = Parameter("a", 1, 1, 4)
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
    a11 = Parameter("a11", 1, 0.0)
    a12 = Parameter("a12", 1, lb_a12, ub_a12)
    a21 = Parameter("a21", 1, lb_a21, ub_a21)
    a22 = Parameter("a22", 1, ub=0.0)
    params = [a11, a12, a21, a22]

    M = Model("lds", params)
    M.set_eps(linear2D_freq)
    q_theta, opt_data, save_path, _ = M.epi(
        mu, num_iters=100, K=1, save_movie_data=True
    )
    g = q_theta.plot_dist()
    M.epi_opt_movie(save_path)

    params = [a11, a12, a21, a22]
    M = Model("lds_2D", params)
    M.set_eps(linear2D_freq)
    q_theta, opt_data, save_path, _ = M.epi(
        mu, num_iters=100, K=1, save_movie_data=True
    )

    q_theta = M.load_epi_dist(mu, k=1)

    M.epi_opt_movie(save_path)
    q_theta, opt_data, save_path, _ = M.epi(
        mu, num_units=31, num_iters=100, K=1, save_movie_data=True
    )
    M.plot_epi_hpsearch(mu)

    opt_data_filename = save_path + "opt_data.csv"

    opt_data_cols = ["k", "iteration", "H", "converged"] + [
        "R%d" % i for i in range(1, M.m + 1)
    ]
    for x, y in zip(opt_data.columns, opt_data_cols):
        assert x == y

    # opt_data_df = pd.read_csv(opt_data_filename)
    # opt_data_df['iteration'] = 2*opt_data_df['iteration']
    # opt_data_df.to_csv(opt_data_filename)
    # with raises(IOError):
    #    M.epi_opt_movie(save_path)
    # os.remove(opt_data_filename)
    # with raises(IOError):
    #    M.epi_opt_movie(save_path)
    assert q_theta is not None
    with raises(ValueError):
        q_theta = M.load_epi_dist(mu, k=20)
    with raises(TypeError):
        q_theta = M.load_epi_dist(mu, k="foo")
    with raises(ValueError):
        q_theta = M.load_epi_dist(mu, k=-1)

    M = Model("foo", params)
    with raises(ValueError):
        q_theta = M.load_epi_dist(mu, k=-1)

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

    # Intentionally swap order in list to insure proper handling.
    params = [a22, a21, a12, a11]
    M = Model("lds2", params)
    M.set_eps(linear2D_freq)
    q_theta, opt_data, save_path, _ = M.epi(
        mu, K=2, num_iters=100, stop_early=True, verbose=True
    )
    with raises(IOError):
        M.epi_opt_movie(save_path)

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

    with raises(ValueError):

        def bad_f(a11, a12, a21, a22):
            return tf.expand_dims(a11 + a12 + a21 + a22, 0)

        M.set_eps(bad_f)

    params = [a22, a21, a12, a11]
    M = Model("lds2", params)
    nf = NormalizingFlow("autoregressive", 4, 1, 2, 10)
    al_hps = AugLagHPs()
    with raises(AttributeError):
        save_path = M.get_save_path(mu, nf, al_hps, None)
    save_path = M.get_save_path(mu, nf, al_hps, eps_name="foo")
    return None


def test_Distribution():
    """ Test Distribution class."""
    tf.random.set_seed(1)
    np.random.seed(1)
    Ds = [2, 4]
    num_dists = 1
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
                init_type, init_params, num_iters=2500, load_if_cached=False, save=False
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
            with raises(TypeError):
                hess_z = q_theta.hessian("foo")


# if __name__ == '__main__':
# test_epi()
