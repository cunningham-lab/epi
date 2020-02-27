""" Test example emergent property statistics functions. """

import numpy as np
import tensorflow as tf
from epi.example_eps import linear2D_freq, linear2D_freq_sq, linear2D_freq_np
import pytest
from pytest import raises

DTYPE = np.float32


def test_linear2D_freq():
    N = 100
    a11s = np.random.normal(0.0, 5.0, (N,1)).astype(DTYPE)
    a12s = np.random.normal(0.0, 5.0, (N,1)).astype(DTYPE)
    a21s = np.random.normal(0.0, 5.0, (N,1)).astype(DTYPE)
    a22s = np.random.normal(0.0, 5.0, (N,1)).astype(DTYPE)
    As = np.concatenate((a11s, a12s, a21s, a22s), axis=1)

    real_lambdas = np.zeros((N,))
    imag_lambdas = np.zeros((N,))

    T_x = linear2D_freq(a11s, a12s, a21s, a22s)
    T_x_sq = linear2D_freq_sq(As)
    for i in range(N):
        real_lambdas[i], imag_lambdas[i] = linear2D_freq_np(a11s[i,0], a12s[i,0], a21s[i,0], a22s[i,0])

    T_x_np = np.stack(
        (
            real_lambdas,
            np.square(real_lambdas - 0.),
            imag_lambdas,
            np.square(imag_lambdas - 2.*np.pi),
        ),
        axis=1,
    )

    assert np.isclose(T_x, T_x_np, rtol=1e-3).all()
    assert np.isclose(T_x_sq, T_x_np, rtol=1e-3).all()

    return None
