from neural_circuits.LRRNN import get_W_eigs_np, get_W_eigs_tf
import numpy as np
import tensorflow as tf


def test_get_W_eigs():
    np.random.seed(0)
    tf.random.set_seed(0)
    N = 15
    r = 2
    U = np.random.uniform(-1.0, 1.0, (N, r)).astype(np.float32)
    V = np.random.uniform(-1.0, 1.0, (N, r)).astype(np.float32)
    U_tf = tf.convert_to_tensor(U[None, :, :])
    V_tf = tf.convert_to_tensor(V[None, :, :])

    g = 0.0
    for K in [1, 5, 100]:
        print("g = %.2f,  K = %d\r" % (g, K), end="")
        _get_W_eigs_np = get_W_eigs_np(g, K)
        _get_W_eigs_tf = get_W_eigs_tf(g, K)
        x = _get_W_eigs_np(U, V)
        T_x = _get_W_eigs_tf(U_tf, V_tf)
        assert np.isclose(x, T_x[0, :2]).all()
    print("")

    gs = [0.01, 0.1, 0.5]
    tols_list = [[1e-1, 1e-2], [1e-1, 1e-2], [1e-1]]
    Ks_list = [[100, 1000], [500, 10000], [5000]]
    for g, tols, Ks in zip(gs, tols_list, Ks_list):
        for tol, K in zip(tols, Ks):
            print("g = %.2f,  K = %d\r" % (g, K), end="")
            _get_W_eigs_np = get_W_eigs_np(g, K)
            _get_W_eigs_tf = get_W_eigs_tf(g, K)
            x = _get_W_eigs_np(U, V)
            T_x = _get_W_eigs_tf(U_tf, V_tf)
            # print('x', x)
            # print('T_x', T_x)
            assert np.isclose(x, T_x[0, :2], atol=tol).all()
    print("")

    M = 200
    print("Testing scaling")
    for g in [0.0, 0.01, 0.5]:
        _get_W_eigs_tf = get_W_eigs_tf(g, 10)
        for N in [25, 100, 250]:
            print("g=%.2f, N=%d\r" % (g, N), end="")
            U = tf.random.uniform((M, N, r), -1.0, 1.0)
            V = tf.random.uniform((M, N, r), -1.0, 1.0)
            T_x = _get_W_eigs_tf(U, V)
            num_infs = tf.reduce_sum(tf.cast(tf.math.is_inf(T_x), tf.float32))
            num_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(T_x), tf.float32))
            assert num_infs + num_nans == 0

    return None


if __name__ == "__main__":
    test_get_W_eigs()
