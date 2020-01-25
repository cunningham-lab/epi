""" Test models. """

import numpy as np
from epi.models import Parameter, Model
from pytest import raises

def test_Parameter_init():
    """Test Parameter initialization."""
    b1 = (-0.1, 1.2)
    p = Parameter('foo', b1)
    assert(p.name == 'foo')
    assert(type(p.bounds) is tuple)
    assert(p.bounds[0] == -0.1)
    assert(p.bounds[1] == 1.2)
    assert(len(p.bounds) == 2)
    
    with raises(TypeError):
        p = Parameter(20, b1)

    p = Parameter('bar', [-0.1, 1.2])
    assert(p.bounds == (-0.1, 1.2))
    p = Parameter('bar', np.array([-0.1, 1.2]))
    assert(p.bounds == (-0.1, 1.2))
    
    with raises(TypeError):
        p = Parameter('foo', 'bar')

    with raises(ValueError):
        p = Parameter('foo', [1, 2, 3])

    with raises(TypeError):
        p = Parameter('foo', ['a', 'b'])
    with raises(TypeError):
        p = Parameter('foo', [1, 'b'])

    with raises(ValueError):
        p = Parameter('foo', [1, -1])

    with raises(ValueError):
        p = Parameter('foo', [1, 1])

    return None

def test_Model_init():
    """Test Model initialization."""
    p1 = Parameter('a', [0, 1])
    p2 = Parameter('b')
    params = [p1, p2]
    M = Model('foo', params)
    assert(M.name == 'foo')
    for i, p in enumerate(M.parameters):
        assert(p == params[i])

    params = [p1, 'bar']
    with raises(TypeError):
        Model('foo', params)

    p3 = Parameter('c', [1, 4])
    p3.bounds = (1, -1)
    params = [p1, p2, p3]
    with raises(ValueError):
        Model('foo', params)

    p3.bounds = (1, 1)
    with raises(ValueError):
        Model('foo', params)

    p3 = Parameter('a', [1, 4])
    params = [p1, p2, p3]
    with raises(ValueError):
        Model('foo', params)

    return None

def test_set_eps():
    params = [Parameter('a'), Parameter('b')]
    M = Model('foo', params)
    with raises(NotImplementedError):
        M._set_eps()
    return None

def test_epi():
    params = [Parameter('a'), Parameter('b')]
    M = Model('foo', params)
    with raises(NotImplementedError):
        M.epi()
    return None

def test_load_epi_dist():
    params = [Parameter('a'), Parameter('b')]
    M = Model('foo', params)
    with raises(NotImplementedError):
        M.load_epi_dist()
    return None

if __name__ == "__main__":
    test_Parameter_init()
    test_Model_init()
