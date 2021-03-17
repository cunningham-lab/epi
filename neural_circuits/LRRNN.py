import numpy as np
import tensorflow as tf

EPS = 1e-6

def get_W_eigs_np(g, K, feed_noise=False):
    def W_eigs(U, V, noise=None):
        U, V = U[None,:,:], V[None,:,:]
        U, V = np.tile(U, [K,1,1]), np.tile(V, [K,1,1])
        if feed_noise:
            U_noise, V_noise = noise
            U = U + g*U_noise
            V = V + g*V_noise
        else:
            U = U + g*np.random.normal(0., 1., U.shape)
            V = V + g*np.random.normal(0., 1., V.shape)
        J = np.matmul(U, np.transpose(V, [0,2,1]))
        Js = (J + np.transpose(J, [0,2,1])) / 2.
        Js_eigs = np.linalg.eigvalsh(Js)
        Js_eig_maxs = np.max(Js_eigs, axis=1)
        Js_eig_max = np.mean(Js_eig_maxs)

        # Take eig of low rank similar mat
        Jr = np.matmul(np.transpose(V,[0,2,1]), U) #+ EPS*np.eye(2)[None,:,:]
        Jr_tr = np.trace(Jr, axis1=1, axis2=2)
        sqrt_term = np.square(Jr_tr) + -4.*np.linalg.det(Jr)
        maybe_complex_term = np.sqrt(np.vectorize(complex)(sqrt_term, 0.))
        J_eig_realmaxs = 0.5 * (Jr_tr + np.real(maybe_complex_term))
        J_eig_realmax = np.mean(J_eig_realmaxs)
        return np.array([J_eig_realmax, Js_eig_max])
    return W_eigs

def get_W_eigs_full_np(g, K, feed_noise=False):
    def W_eigs(U, V, noise=None):
        J = np.matmul(U, np.transpose(V))[None, :, :]
        J = np.tile(J, [K,1,1])
        if feed_noise:
            J = J + g*noise
        else:
            J = J + g*np.random.normal(0., 1., J.shape)
        Js = (J + np.transpose(J, [0,2,1])) / 2.
        Js_eigs = np.linalg.eigvalsh(Js)
        Js_eig_maxs = np.max(Js_eigs, axis=1)
        Js_eig_max = np.mean(Js_eig_maxs)
       
        J_eig_realmaxs = []
        for k in range(K):
            _J = J[k]
            w,v = np.linalg.eig(_J)
            J_eig_realmaxs.append(np.max(np.real(w)))
        J_eig_realmax = np.mean(J_eig_realmaxs)
        return np.array([J_eig_realmax, Js_eig_max])
    return W_eigs

def get_W_eigs_tf(g, K, Js_eig_max_mean=1.5, J_eig_realmax_mean=0.5, feed_noise=False):
    def W_eigs(U, V, noise=None):
        U, V = U[:,None,:,:], V[:,None,:,:]
        U, V = tf.tile(U, [1,K,1,1]), tf.tile(V, [1,K,1,1])
        if feed_noise:
            U_noise, V_noise = noise
            U = U + g*U_noise
            V = V + g*V_noise
        else:
            U = U + g*tf.random.normal(U.shape, 0., 1.)
            V = V + g*tf.random.normal(V.shape, 0., 1.)
        J = tf.matmul(U, tf.transpose(V, [0,1,3,2]))
        Js = (J + tf.transpose(J, [0,1,3,2])) / 2.
        Js_eigs = tf.linalg.eigvalsh(Js)
        Js_eig_maxs = tf.reduce_max(Js_eigs, axis=2)
        Js_eig_max = tf.reduce_mean(Js_eig_maxs, axis=1)

        # Take eig of low rank similar mat
        Jr = tf.matmul(tf.transpose(V, [0,1,3,2]), U) + EPS*tf.eye(2)[None,None,:,:]
        Jr_tr = tf.linalg.trace(Jr)
        maybe_complex_term = tf.sqrt(tf.complex(tf.square(Jr_tr) + -4.*tf.linalg.det(Jr), 0.))
        J_eig_realmaxs = 0.5 * (Jr_tr + tf.math.real(maybe_complex_term))
        J_eig_realmax = tf.reduce_mean(J_eig_realmaxs, axis=1)

        T_x = tf.stack([J_eig_realmax, Js_eig_max,
                        tf.square(J_eig_realmax-J_eig_realmax_mean),
                        tf.square(Js_eig_max-Js_eig_max_mean)], axis=1)
        return T_x
    return W_eigs

def get_W_eigs_full_tf(g, K, Js_eig_max_mean=1.5, J_eig_realmax_mean=0.5, feed_noise=False):
    def W_eigs(U, V, noise=None):
        J = tf.matmul(U, tf.transpose(V, [0,2,1]))
        J = tf.tile(J[:,None,:,:], (1,K,1,1))
        if feed_noise:
            J = J + g*noise
        else:
            J = J + g*tf.random.normal(J.shape, 0., 1.)
        Js = (J + tf.transpose(J, [0,1,3,2])) / 2.
        Js_eigs = tf.linalg.eigvalsh(Js)
        Js_eig_maxs = tf.reduce_max(Js_eigs, axis=2)
        Js_eig_max = tf.reduce_mean(Js_eig_maxs, axis=1)

        # Take eig of low rank similar mat
        Jr = tf.matmul(tf.transpose(V, [0,2,1]), U) #+ EPS*tf.eye(2)[None,None,:,:]
        Jr_tr = tf.linalg.trace(Jr)
        maybe_complex_term = tf.sqrt(tf.complex(tf.square(Jr_tr) + -4.*tf.linalg.det(Jr), 0.))
        J_eig_realmax = 0.5 * (Jr_tr + tf.math.real(maybe_complex_term))
        J_eig_realmax = tf.math.maximum(J_eig_realmax, g)

        T_x = tf.stack([J_eig_realmax, Js_eig_max,
                        tf.square(J_eig_realmax-J_eig_realmax_mean),
                        tf.square(Js_eig_max-Js_eig_max_mean)], axis=1)
        return T_x
    return W_eigs

