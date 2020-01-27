""" Test normalizing flow architectures. """

import numpy as np
from epi.normalizing_flows import Architecture
from pytest import raises


def test_Architecture_init():
    """Test architecture initialization."""
    arch_type = "coupling"
    D = 4
    num_stages = 1
    num_layers = 2
    num_units = 15

    # Check setters.
    A = Architecture(arch_type, D, num_stages, num_layers, num_units)
    assert A.arch_type == "coupling"
    assert A.D == D
    assert A.num_stages == num_stages
    assert A.num_layers == num_layers
    assert A.num_units == num_units
    assert A.batch_norm
    assert A.post_affine
    assert A.lb is None
    assert A.ub is None
    assert A.random_seed == 1

    # Test autoregressive
    A = Architecture('autoregressive', D, num_stages, num_layers, num_units)
    assert A.arch_type == "autoregressive"

    lb = -2.*np.ones((D,))
    ub = 2.*np.ones((D,))
    bounds = (lb, ub)
    A = Architecture(arch_type, D, num_stages, num_layers, num_units, False, False, bounds, 5)
    assert not A.batch_norm
    assert not A.post_affine
    assert np.equal(A.lb, lb).all()
    assert np.equal(A.ub, ub).all()
    assert A.random_seed == 5

    with raises(TypeError):
        A = Architecture(0, D, num_stages, num_layers, num_units)
    with raises(ValueError):
        A = Architecture('foo', D, num_stages, num_layers, num_units)

    with raises(TypeError):
        A = Architecture(arch_type, 2., num_stages, num_layers, num_units)
    with raises(ValueError):
        A = Architecture(arch_type, 1, num_stages, num_layers, num_units)

    with raises(TypeError):
        A = Architecture(arch_type, D, 2., num_layers, num_units)
    with raises(ValueError):
        A = Architecture(arch_type, D, 0, num_layers, num_units)

    with raises(TypeError):
        A = Architecture(arch_type, D, num_stages, 2., num_units)
    with raises(ValueError):
        A = Architecture(arch_type, D, num_stages, 0, num_units)

    with raises(TypeError):
        A = Architecture(arch_type, D, num_stages, num_layers, 2.)
    with raises(ValueError):
        A = Architecture(arch_type, D, num_stages, num_layers, 0)

    with raises(TypeError):
        A = Architecture(arch_type, D, num_stages, num_layers, 2.)
    with raises(ValueError):
        A = Architecture(arch_type, D, num_stages, num_layers, 0)

    with raises(TypeError):
        A = Architecture(arch_type, D, num_stages, num_layers, num_units, batch_norm=1.)

    with raises(TypeError):
        A = Architecture(arch_type, D, num_stages, num_layers, num_units, post_affine=1.)

    with raises(TypeError):
        A = Architecture(arch_type, D, num_stages, num_layers, num_units, randon_seed=1.)

    return None

"""
def test_CouplingArch_init():
    CA = CouplingArch(4, 1, 2, 10)
    assert CA.D == 4
    assert CA.num_couplings == 1
    assert CA.num_layers == 2
    assert CA.num_units == 10

    with raises(TypeError):
        CA = CouplingArch("a", 1, 2, 10)
    with raises(ValueError):
        CA = CouplingArch(1, 1, 2, 10)

    with raises(TypeError):
        CA = CouplingArch(4, "a", 2, 10)
    with raises(ValueError):
        CA = CouplingArch(4, 0, 2, 10)

    with raises(TypeError):
        CA = CouplingArch(4, 1, "a", 10)
    with raises(ValueError):
        CA = CouplingArch(4, 1, 0, 10)

    with raises(TypeError):
        CA = CouplingArch(4, 1, 2, "a")
    with raises(ValueError):
        CA = CouplingArch(4, 1, 2, 0)

    CA = CouplingArch(4, 2, 2, 10)
    x, log_q_x = CA(100)
    return None


def test_to_string():
    A = Arch("planar", 2, True)
    assert A.to_string() == "2_planar_PA"

    A = Arch("planar", 100, True)
    assert A.to_string() == "100_planar_PA"

    A = Arch("planar", 2, False)
    assert A.to_string() == "2_planar"

    CA = CouplingArch(4, 1, 2, 10)
    assert CA.to_string() == "NVP_1C_2L_10U_PA"

    CA = CouplingArch(4, 1, 2, 10, False)
    assert CA.to_string() == "NVP_1C_2L_10U"

    return None
"""

if __name__ == "__main__":
    test_Arch_init()
