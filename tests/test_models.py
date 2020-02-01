""" Test models. """

import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
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
    mu = np.array([2*np.pi, 0.1*np.pi])
    def lds_freq(a11, a12, a21, a22):
        tau = 1.
        c11 = a11 / tau
        c12 = a12 / tau
        c21 = a21 / tau
        c22 = a22 / tau

        beta = tf.complex(tf.square(c11 + c22) - 4.*(c11*c22 - c12*c21), np.float32(0.))
        beta_sqrt = tf.sqrt(beta)
        lambda_imag = tf.math.imag(0.5*beta_sqrt)
        freq = 2.*np.pi*lambda_imag
        T_x = tf.concat((freq, tf.square(freq-mu[0])), axis=1)

        return T_x

    bounds = [-10, 10]
    a11 = Parameter("a11", bounds)
    a12 = Parameter("a12", bounds)
    a21 = Parameter("a21", bounds)
    a22 = Parameter("a22", bounds)
    params = [a11, a12, a21, a22]
    M = Model("LDS", params)

    M.set_eps(lds_freq, 2)
    M.epi(mu, num_iters=500)

    return None


