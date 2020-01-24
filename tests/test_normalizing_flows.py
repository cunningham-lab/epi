""" Test normalizing flow architectures. """

from epi.normalizing_flows import Arch
from pytest import raises

def test_Arch_init():
    """Test architecture initialization."""
    arch_type = "planar"
    num_layers = 2
    post_affine = True

    # Check setters.
    A = Arch(arch_type, num_layers, post_affine)
    assert(A.arch_type == 'planar')
    assert(A.num_layers == 2)
    assert(A.post_affine)

    # Check type checking.
    with raises(TypeError):
        A = Arch(0, num_layers, post_affine) 
    with raises(TypeError):
        A = Arch({}, num_layers, post_affine) 

    with raises(TypeError):
        A = Arch(arch_type, 2., post_affine) 
    with raises(TypeError):
        A = Arch(arch_type, 'foo', post_affine) 
    with raises(TypeError):
        A = Arch(arch_type, {}, post_affine) 

    with raises(TypeError):
        A = Arch(arch_type, num_layers, 0) 
    with raises(TypeError):
        A = Arch(arch_type, num_layers, 'False') 

    # Check value checking
    with raises(ValueError):
        A = Arch('ar', num_layers, post_affine) 
    with raises(ValueError):
        A = Arch('coupling', num_layers, post_affine) 
    with raises(ValueError):
        A = Arch('foo', num_layers, post_affine) 

    with raises(ValueError):
        A = Arch(arch_type, 0, post_affine) 
    with raises(ValueError):
        A = Arch(arch_type, -100, post_affine) 

    return None
        
def test_to_string():
    """Test string construction from architecture parameters."""
    A = Arch('planar', 2, True)
    assert(A.to_string() == '2_planar_PA')
    
    A = Arch('planar', 100, True)
    assert(A.to_string() == '100_planar_PA')
        
    A = Arch('planar', 2, False)
    assert(A.to_string() == '2_planar')

    return None

if __name__ == '__main__':
    test_Arch_init()
    test_to_string()
