""" Test normalizing flow architectures. """

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from epi.normalizing_flows import NormalizingFlow, IntervalFlow
from epi.util import pairplot


def plot_nf(nf):
    M = 10000
    z, log_q_z = nf(M)
    D = z.shape[1]
    keep = log_q_z > -1.0
    # z, log_q_z = z.numpy(), log_q_z.numpy()
    z, log_q_z = z.numpy()[keep], log_q_z.numpy()[keep]
    labels = ["z%d" % (d + 1) for d in range(D)]
    figsize = (6, 6)
    # pairplot(z, range(D), labels=labels, c=log_q_z, figsize=figsize)
    pairplot(z, range(D), labels=labels, c=log_q_z, figsize=figsize, lb=nf.lb, ub=nf.ub)
    plt.show(True)
    return None


def test_boundaries():
    D = 2
    lb = -1 * np.ones((D,))
    ub = 1 * np.ones((D,))
    nf = NormalizingFlow(
        "coupling",
        D,
        2,
        2,
        25,
        batch_norm=False,
        elemwise_fn="spline",
        num_bins=32,
        bounds=(lb, ub),
        post_affine=False,
        random_seed=2,
    )
    plot_nf(nf)
    _M = 10
    _x = np.random.normal(0.0, 1.0, (_M, D // 2))
    # print('bin widths')
    # print(tf.reduce_sum(nf.bijector_fns[0](_x), axis=2))

    mu = 0.75 * np.ones((D,))
    Sigma = 0.25 * np.eye(D)
    # Sigma = 1.*np.eye(D)
    # Sigma[0,1] = 0.75
    # Sigma[1,0] = 0.75
    nf.initialize(mu, Sigma, num_iters=int(5e3), log_rate=200, verbose=True)

    # print('bin widths')
    # print(tf.reduce_sum(nf.bijector_fns[0](_x), axis=2))

    plot_nf(nf)

    return None


if __name__ == "__main__":
    test_boundaries()
