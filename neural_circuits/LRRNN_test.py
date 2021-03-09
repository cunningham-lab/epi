from neural_circuits.LRRNN import get_W_eigs_np, get_W_eigs_tf
import numpy as np
import tensorflow as tf

def test_get_W_eigs():
    N = 15
    r = 2
    U = np.random.uniform(-1., 1., (N,r)).astype(np.float32)
    V = np.random.uniform(-1., 1., (N,r)).astype(np.float32)
    U_tf = tf.convert_to_tensor(U[None,:,:])
    V_tf = tf.convert_to_tensor(V[None,:,:])

    g = 0.
    for K in [1, 5, 100]:
        print('g = %.2f,  K = %d' % (g, K))
        _get_W_eigs_np = get_W_eigs_np(g, K)
        _get_W_eigs_tf = get_W_eigs_tf(g, K)
        x = _get_W_eigs_np(U,V)
        T_x = _get_W_eigs_tf(U_tf, V_tf)
        assert(np.isclose(x, T_x[0,:2]).all())

    gs = [0.01, 0.1, 0.5]
    tols_list = [[1e-1, 1e-2], [1e-1, 1e-2], [1e-1]]
    Ks_list = [[100, 1000], [500, 2000], [5000]]
    for g, tols, Ks in zip(gs, tols_list, Ks_list):
        for tol, K in zip(tols, Ks):
            print('g = %.2f,  K = %d' % (g, K))
            _get_W_eigs_np = get_W_eigs_np(g, K)
            _get_W_eigs_tf = get_W_eigs_tf(g, K)
            x = _get_W_eigs_np(U,V)
            print('x', x)
            T_x = _get_W_eigs_tf(U_tf, V_tf)
            print('T_x', T_x[0,:2])
            assert(np.isclose(x, T_x[0,:2], atol=tol).all())

    return None

if __name__ == "__main__":
    test_get_W_eigs()
