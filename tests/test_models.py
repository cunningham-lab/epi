""" Test models. """

import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.example_eps import linear2D_freq
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

    params = [p1, p2]
    M = Model("foo", params)
    with raises(NotImplementedError):
        M.load_epi_dist()

    return None


def test_epi():
    mu = np.array([0., 0.1, 2*np.pi, 0.1*np.pi])

    lb_a12 = 0.
    ub_a12 = 10.
    lb_a21 = -10.
    ub_a21 = 0.
    a11 = Parameter("a11", [0., np.PINF])
    a12 = Parameter("a12", [lb_a12, ub_a12])
    a21 = Parameter("a21", [lb_a21, ub_a21])
    a22 = Parameter("a22", [np.NINF, 0.])
    params = [a11, a12, a21, a22]

    M = Model("lds", params)
    M.set_eps(linear2D_freq, 4)
    q_theta = M.epi(mu, num_iters=100)

    z, log_q_z = q_theta(1000)
    assert np.sum(z[:,0] < 0.) == 0
    assert np.sum(z[:,1] < lb_a12) == 0
    assert np.sum(z[:,1] > ub_a12) == 0
    assert np.sum(z[:,2] < lb_a21) == 0
    assert np.sum(z[:,2] > ub_a21) == 0
    assert np.sum(z[:,3] > 0.) == 0
    assert np.sum(1 - np.isfinite(z)) == 0
    assert np.sum(1 - np.isfinite(log_q_z)) == 0


    # Intentionally swap order in list to insure proper handling.
    params = [a22, a21, a12, a11]
    M = Model("lds2", params)
    M.set_eps(linear2D_freq, 4)
    q_theta = M.epi(mu, num_iters=100)

    z, log_q_z = q_theta(1000)
    assert np.sum(z[:,0] < 0.) == 0
    assert np.sum(z[:,1] < lb_a12) == 0
    assert np.sum(z[:,1] > ub_a12) == 0
    assert np.sum(z[:,2] < lb_a21) == 0
    assert np.sum(z[:,2] > ub_a21) == 0
    assert np.sum(z[:,3] > 0.) == 0
    assert np.sum(1 - np.isfinite(z)) == 0
    assert np.sum(1 - np.isfinite(log_q_z)) == 0

    return None


