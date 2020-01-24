""" Test normalizing flow architectures. """

from epi.normalizing_flows import Arch, CouplingArch
from pytest import raises


def test_Arch_init():
    """Test architecture initialization."""
    arch_type = "planar"
    num_layers = 2
    post_affine = True

    # Check setters.
    A = Arch(arch_type, num_layers, post_affine)
    assert A.arch_type == "planar"
    assert A.num_layers == 2
    assert A.post_affine

    # Check type checking.
    with raises(TypeError):
        A = Arch(0, num_layers, post_affine)
    with raises(TypeError):
        A = Arch({}, num_layers, post_affine)

    with raises(TypeError):
        A = Arch(arch_type, 2.0, post_affine)
    with raises(TypeError):
        A = Arch(arch_type, "foo", post_affine)
    with raises(TypeError):
        A = Arch(arch_type, {}, post_affine)

    with raises(TypeError):
        A = Arch(arch_type, num_layers, 0)
    with raises(TypeError):
        A = Arch(arch_type, num_layers, "False")

    # Check value checking
    with raises(ValueError):
        A = Arch("ar", num_layers, post_affine)
    with raises(ValueError):
        A = Arch("coupling", num_layers, post_affine)
    with raises(ValueError):
        A = Arch("foo", num_layers, post_affine)

    with raises(ValueError):
        A = Arch(arch_type, 0, post_affine)
    with raises(ValueError):
        A = Arch(arch_type, -100, post_affine)

    return None


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
    """Test string construction from architecture parameters."""
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


def test_to_model():
    """Test tensorflow model construction from architecture parameters."""
    A = Arch("planar", 2, False)
    with raises(NotImplementedError):
        A.to_model()
    return None


if __name__ == "__main__":
    test_Arch_init()
    test_CouplingArch_init()
    test_to_string()
    test_to_string()
