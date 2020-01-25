""" Test models. """

import numpy as np
from epi.models import Parameter
from pytest import raises

def test_Parameter_init():
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



if __name__ == "__main__":
    test_Parameter_init()
