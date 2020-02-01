""" Test example emergent property statistics functions. """

import numpy as np
import tensorflow as tf
from epi.example_eps import linear2D_freq
import pytest
from pytest import raises

DTYPE = np.float32


def linear2D_freq_np(a11, a12, a21, a22):
    tau = 1.0
    C = np.array([[a11, a12], [a21, a22]]) / tau
    eigs = np.linalg.eigvals(C)
    eig1, eig2 = eigs[0], eigs[1]
    eig1_r = np.real(eig1)
    eig1_i = np.imag(eig1)
    eig2_r = np.real(eig2)
    eig2_i = np.imag(eig2)
    if eig1_r >= eig2_r:
        alpha = eig1_r
    else:
        alpha = eig2_r

    if eig1_i >= eig2_i:
        omega = 2 * np.pi * eig1_i
    else:
        omega = 2 * np.pi * eig2_i

    return alpha, omega


def test_linear2D_freq():
    N = 100
    a11s = np.random.normal(0.0, 5.0, (N,)).astype(DTYPE)
    a12s = np.random.normal(0.0, 5.0, (N,)).astype(DTYPE)
    a21s = np.random.normal(0.0, 5.0, (N,)).astype(DTYPE)
    a22s = np.random.normal(0.0, 5.0, (N,)).astype(DTYPE)

    alphas = np.zeros((N,))
    omegas = np.zeros((N,))

    T_x = linear2D_freq(a11s, a12s, a21s, a22s)
    for i in range(N):
        alphas[i], omegas[i] = linear2D_freq_np(a11s[i], a12s[i], a21s[i], a22s[i])

    mean_alphas = np.mean(alphas)
    mean_omegas = np.mean(omegas)

    T_x_np = np.stack(
        (
            alphas,
            np.square(alphas - mean_alphas),
            omegas,
            np.square(omegas - mean_omegas),
        ),
        axis=1,
    )

    assert np.isclose(T_x, T_x_np, rtol=1e-3).all()

    return None
