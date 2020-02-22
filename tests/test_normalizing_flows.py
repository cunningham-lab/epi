""" Test normalizing flow architectures. """

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from epi.normalizing_flows import NormalizingFlow, IntervalFlow
from pytest import raises

EPS = 1e-6


def test_NormalizingFlow_init():
    """Test architecture initialization."""
    arch_type = "coupling"
    D = 4
    num_stages = 1
    num_layers = 2
    num_units = 15

    tf.random.set_seed(0)
    np.random.seed(0)

    # Check setters.
    nf = NormalizingFlow(arch_type, D, num_stages, num_layers, num_units)
    assert nf.arch_type == "coupling"
    assert nf.D == D
    assert nf.num_stages == num_stages
    assert nf.num_layers == num_layers
    assert nf.num_units == num_units
    assert nf.batch_norm
    assert nf.post_affine
    assert nf.lb is None
    assert nf.ub is None
    assert nf.random_seed == 1

    # Test autoregressive
    nf = NormalizingFlow("autoregressive", D, num_stages, num_layers, num_units)
    assert nf.arch_type == "autoregressive"

    lb = -2.0 * np.ones((D,))
    ub = 2.0 * np.ones((D,))
    bounds = (lb, ub)
    nf = NormalizingFlow(
        arch_type, D, num_stages, num_layers, num_units, False, None, False, bounds, 5
    )
    assert not nf.batch_norm
    assert not nf.post_affine
    assert np.equal(nf.lb, lb).all()
    assert np.equal(nf.ub, ub).all()
    assert nf.random_seed == 5
    nf = NormalizingFlow(
        arch_type, D, num_stages, num_layers, num_units, False, None, False, [lb, ub], 5
    )
    assert np.equal(nf.lb, lb).all()
    assert np.equal(nf.ub, ub).all()

    # Test error handling.
    with raises(TypeError):
        nf = NormalizingFlow(0, D, num_stages, num_layers, num_units)
    with raises(ValueError):
        nf = NormalizingFlow("foo", D, num_stages, num_layers, num_units)

    with raises(TypeError):
        nf = NormalizingFlow(arch_type, 2.0, num_stages, num_layers, num_units)
    with raises(ValueError):
        nf = NormalizingFlow(arch_type, 1, num_stages, num_layers, num_units)

    with raises(TypeError):
        nf = NormalizingFlow(arch_type, D, 2.0, num_layers, num_units)
    with raises(ValueError):
        nf = NormalizingFlow(arch_type, D, 0, num_layers, num_units)

    with raises(TypeError):
        nf = NormalizingFlow(arch_type, D, num_stages, 2.0, num_units)
    with raises(ValueError):
        nf = NormalizingFlow(arch_type, D, num_stages, 0, num_units)

    with raises(TypeError):
        nf = NormalizingFlow(arch_type, D, num_stages, num_layers, 2.0)
    with raises(ValueError):
        nf = NormalizingFlow(arch_type, D, num_stages, num_layers, 0)

    with raises(TypeError):
        nf = NormalizingFlow(arch_type, D, num_stages, num_layers, 2.0)
    with raises(ValueError):
        nf = NormalizingFlow(arch_type, D, num_stages, num_layers, 0)

    with raises(TypeError):
        nf = NormalizingFlow(
            arch_type, D, num_stages, num_layers, num_units, batch_norm=1.0
        )

    with raises(TypeError):
        nf = NormalizingFlow(
            arch_type,
            D,
            num_stages,
            num_layers,
            num_units,
            batch_norm=True,
            bn_momentum="foo",
        )
    with raises(ValueError):
        nf = NormalizingFlow(
            arch_type, D, num_stages, num_layers, num_units, bounds=(lb, ub, ub)
        )
    with raises(TypeError):
        nf = NormalizingFlow(
            arch_type, D, num_stages, num_layers, num_units, bounds=("foo", "bar")
        )

    with raises(TypeError):
        nf = NormalizingFlow(
            arch_type, D, num_stages, num_layers, num_units, bounds="foo"
        )

    with raises(TypeError):
        nf = NormalizingFlow(
            arch_type, D, num_stages, num_layers, num_units, random_seed=1.0
        )

    # Check that q0 has correct statistics
    nf = NormalizingFlow(arch_type, D, num_stages, num_layers, num_units)
    z = nf.q0.sample(100000).numpy()
    print(np.mean(z, 0))
    assert np.isclose(np.mean(z, 0), np.zeros((D,)), atol=1e-2).all()
    assert np.isclose(np.cov(z.T), np.eye(D), atol=1e-1).all()

    return None


def test_NormalizingFlow_call():
    D = 4
    num_stages = 1
    num_layers = 2
    num_units = 15
    N = 100
    # Check that
    arch_types = ["autoregressive", "coupling"]
    stage_bijectors = [tfp.bijectors.MaskedAutoregressiveFlow, tfp.bijectors.RealNVP]
    for arch_type, stage_bijector in zip(arch_types, stage_bijectors):
        nf = NormalizingFlow(arch_type, D, num_stages, num_layers, num_units)
        z = nf(N)
        bijectors = nf.trans_dist.bijector.bijectors
        assert type(bijectors[1]) is stage_bijector
        assert type(bijectors[0]) is tfp.bijectors.Chain

        nf = NormalizingFlow(arch_type, D, 2, num_layers, num_units)
        z = nf(N)
        bijectors = nf.trans_dist.bijector.bijectors
        assert type(bijectors[4]) is stage_bijector
        assert type(bijectors[3]) is tfp.bijectors.Permute
        assert type(bijectors[2]) is tfp.bijectors.BatchNormalization
        assert type(bijectors[1]) is stage_bijector
        assert type(bijectors[0]) is tfp.bijectors.Chain

        nf = NormalizingFlow(arch_type, D, 3, num_layers, num_units)
        z = nf(N)
        bijectors = nf.trans_dist.bijector.bijectors
        assert type(bijectors[7]) is stage_bijector
        assert type(bijectors[6]) is tfp.bijectors.Permute
        assert type(bijectors[5]) is tfp.bijectors.BatchNormalization
        assert type(bijectors[4]) is stage_bijector
        assert type(bijectors[3]) is tfp.bijectors.Permute
        assert type(bijectors[2]) is tfp.bijectors.BatchNormalization
        assert type(bijectors[1]) is stage_bijector
        assert type(bijectors[0]) is tfp.bijectors.Chain

    return None


def test_to_string():
    nf = NormalizingFlow("coupling", 4, 1, 2, 15)
    #assert nf.to_string() == "D4_C1_L2_U15_bnmom=9.90E-01_PA_rs1"
    #nf = NormalizingFlow("coupling", 100, 2, 4, 200, random_seed=20)
    #assert nf.to_string() == "D100_C2_L4_U200_bnmom=9.90E-01_PA_rs20"
    #nf = NormalizingFlow("coupling", 4, 1, 2, 15, bn_momentum=0.999, post_affine=False)
    #assert nf.to_string() == "D4_C1_L2_U15_bnmom=9.99E-01_rs1"
    nf = NormalizingFlow(
        "autoregressive", 4, 1, 2, 15, batch_norm=False, post_affine=False
    )
    assert nf.to_string() == "D4_AR1_L2_U15_rs1"
    nf = NormalizingFlow(
        "autoregressive", 4, 4, 2, 15, batch_norm=False, post_affine=False
    )
    assert nf.to_string() == "D4_AR4_L2_U15_rs1"


def interval_flow_np(x, lb, ub):
    def softplus(x):
        return np.log(1 + np.exp(-np.abs(x))) + max(0.0, x)

    D = x.shape[0]
    y = np.zeros((D,))
    ldj = 0.0
    for i in range(D):
        x_i = x[i]
        lb_i = lb[i]
        ub_i = ub[i]
        has_lb = not np.isneginf(lb_i)
        has_ub = not np.isposinf(ub_i)
        if has_lb and has_ub:
            m = (ub_i - lb_i) / 2
            c = (ub_i + lb_i) / 2
            y[i] = m * np.tanh(x_i) + c
            ldj += np.log(m) + np.log(1.0 - np.square(np.tanh(x_i)) + EPS)
        elif has_lb:
            y[i] = softplus(x_i) + lb_i
            ldj += np.log(1.0 / (1.0 + np.exp(-x_i)) + EPS)
        elif has_ub:
            y[i] = -softplus(x_i) + ub_i
            ldj += x_i - softplus(x_i)
        else:
            y[i] = x_i

    return y, ldj


def test_interval_flow():
    N = 10
    Ds = [2, 4, 10, 15]
    rtol = 1e-1

    np.random.seed(0)
    tf.random.set_seed(0)

    lb = np.array([float("-inf"), float("-inf"), -100.0, 20.0])
    ub = np.array([float("inf"), 100.0, 30.0, float("inf")])
    IF = IntervalFlow(lb, ub)
    x = np.random.normal(0.0, 2.0, (N, 4)).astype(np.float32)
    y, ldj = IF.forward_and_log_det_jacobian(tf.constant(x))
    x_inv = IF.inverse(y)
    ildj = IF.inverse_log_det_jacobian(y, 1)
    assert np.isclose(x_inv, x, rtol=rtol).all()
    assert np.isclose(ldj, -ildj, rtol=rtol).all()
    for i in range(N):
        y_np, ldj_np = interval_flow_np(x[i], lb, ub)
        assert np.isclose(y[i], y_np, rtol=rtol).all()
        assert np.isclose(ldj[i], ldj_np, rtol=rtol)

    for D in Ds:
        lb = np.array(D * [float("-inf")])
        ub = np.array(D * [float("inf")])
        IF = IntervalFlow(lb, ub)
        x = np.random.normal(0.0, 10.0, (N, D)).astype(np.float32)
        y, ldj = IF.forward_and_log_det_jacobian(tf.constant(x))
        x_inv = IF.inverse(y)
        ildj = IF.inverse_log_det_jacobian(y, 1)
        assert np.isclose(x_inv, x, rtol=rtol).all()
        assert np.isclose(ldj, -ildj, rtol=rtol).all()
        for i in range(N):
            y_np, ldj_np = interval_flow_np(x[i], lb, ub)
            assert np.isclose(y[i], y_np, rtol=rtol).all()
            assert np.isclose(ldj[i], ldj_np, rtol=rtol)

        lb = np.random.uniform(-1000, 1000, (D,))
        ub = np.array(D * [float("inf")])
        IF = IntervalFlow(lb, ub)
        x = np.random.normal(0.0, 3.0, (N, D)).astype(np.float32)
        y, ldj = IF.forward_and_log_det_jacobian(tf.constant(x))
        x_inv = IF.inverse(y)
        ildj = IF.inverse_log_det_jacobian(y, 1)
        assert np.isclose(x_inv, x, rtol=rtol).all()
        assert np.isclose(ldj, -ildj, rtol=rtol).all()
        for i in range(N):
            y_np, ldj_np = interval_flow_np(x[i], lb, ub)
            assert np.isclose(y[i], y_np, rtol=rtol).all()
            assert np.isclose(ldj[i], ldj_np, rtol=rtol)

        lb = np.array(D * [float("-inf")])
        ub = np.random.uniform(-1000, 1000, (D,))
        IF = IntervalFlow(lb, ub)
        x = np.random.normal(0.0, 3.0, (N, D)).astype(np.float32)
        y, ldj = IF.forward_and_log_det_jacobian(tf.constant(x))
        x_inv = IF.inverse(y)
        ildj = IF.inverse_log_det_jacobian(y, 1)
        assert np.isclose(x_inv, x, rtol=rtol).all()
        assert np.isclose(ldj, -ildj, rtol=rtol).all()
        for i in range(N):
            y_np, ldj_np = interval_flow_np(x[i], lb, ub)
            assert np.isclose(y[i], y_np, rtol=rtol).all()
            assert np.isclose(ldj[i], ldj_np, rtol=rtol)

        lb = np.random.uniform(-10, -1, (D,))
        ub = np.random.uniform(1, 10, (D,))
        IF = IntervalFlow(lb, ub)
        x = np.random.normal(0.0, 2.0, (N, D)).astype(np.float32)
        y, ldj = IF.forward_and_log_det_jacobian(tf.constant(x))
        x_inv = IF.inverse(y)
        ildj = IF.inverse_log_det_jacobian(y, 1)
        assert np.isclose(x_inv, x, rtol=rtol).all()
        assert np.isclose(ldj, -ildj, rtol=rtol).all()
        for i in range(N):
            y_np, ldj_np = interval_flow_np(x[i], lb, ub)
            assert np.isclose(y[i], y_np, rtol=rtol).all()
            assert np.isclose(ldj[i], ldj_np, rtol=rtol)

    with raises(TypeError):
        IF = IntervalFlow("foo", ub)
    with raises(TypeError):
        IF = IntervalFlow(lb, "foo")

    with raises(ValueError):
        IF = IntervalFlow(lb, ub[:3])

    tmp = ub[2]
    ub[2] = lb[2]
    lb[2] = tmp
    with raises(ValueError):
        IF = IntervalFlow(lb, ub)

    D = 2
    lb = [0.0, -1.0]
    ub = [1.0, 0.0]
    IF = IntervalFlow(lb, ub)
    x = np.random.normal(0.0, 1.0, (N, D)).astype(np.float32)
    y, ldj = IF.forward_and_log_det_jacobian(tf.constant(x))
    x_inv = IF.inverse(y)
    ildj = IF.inverse_log_det_jacobian(y, 1)
    assert np.isclose(x_inv, x, rtol=rtol).all()
    assert np.isclose(ldj, -ildj, rtol=rtol).all()
    for i in range(N):
        y_np, ldj_np = interval_flow_np(x[i], lb, ub)
        assert np.isclose(y[i], y_np, rtol=rtol).all()

    return None


def test_initialization():
    D = 4
    nf = NormalizingFlow(
        "autoregressive", D, 2, 2, 15, batch_norm=True, post_affine=True
    )
    init_type = "iso_gauss"
    loc = -0.5
    scale = 2.0
    init_params = {"loc": loc, "scale": scale}
    nf.initialize(init_type, init_params)

    z = nf.sample(int(1e4))
    z = z.numpy()
    mean_z = np.mean(z, 0)
    Sigma_z = np.cov(z.T)
    assert np.isclose(mean_z, loc * np.ones((D,)), atol=1e-1).all()
    assert np.isclose(Sigma_z, scale * np.eye(D), atol=1e-1).all()

    # For init load
    nf.initialize(init_type, init_params)

    # Bounds
    lb = np.zeros((D,))
    ub = np.ones((D,))
    nf = NormalizingFlow(
        "autoregressive", D, 2, 2, 15, batch_norm=True, bounds=(lb, ub)
    )
    nf.initialize(init_type, init_params)

    return None


if __name__ == "__main__":
    test_to_string()
