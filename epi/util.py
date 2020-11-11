""" General util functions for EPI. """

import numpy as np
import tensorflow as tf
import pickle
import os
import hashlib
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KernelDensity
from epi.error_formatters import format_type_err_msg

def dbg_check(tensor, name):
    num_elems = 1
    for dim in tensor.shape:
        num_elems *= dim
    num_infs = tf.reduce_sum(tf.cast(tf.math.is_inf(tensor), tf.float32))
    num_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(tensor), tf.float32))

    print(name, "infs %d/%d" % (num_infs, num_elems), "nans %d/%d" % (num_nans, num_elems))
    return num_nans or num_infs

def get_hash(hash_vars):
    m = hashlib.md5()
    for hash_var in hash_vars:
        if hash_var is None:
            continue
        elif type(hash_var) is str:
            hash_var = hash_var.encode('utf-8')
        m.update(hash_var)
    return m.hexdigest()

def set_dir_index(index, index_file):
    exists = os.path.exists(index_file)
    if exists:
        with open(index_file, "rb") as f:
            cur_index = pickle.load(f)
        for key, value in cur_index.items():
            if type(value) is np.ndarray:
                assert(np.isclose(index[key], value).all())
            else:
                assert(index[key] == value)
    else:
        with open(index_file, "wb") as f:
            pickle.dump(index, f)
    return exists

def get_dir_index(path):
    try:
        with open(path, "rb") as f:
            index = pickle.load(f)
    except FileNotFoundError:
        return None
    return index

def gaussian_backward_mapping(mu, Sigma):
    """Calculates natural parameter of multivaraite gaussian from mean and cov.

    :param mu: Mean of gaussian
    :type mu: np.ndarray
    :param Sigma: Covariance of gaussian.
    :type Sigma: np.ndarray
    :return: Natural parameter of gaussian.
    :rtype: np.ndarray
    """
    if type(mu) is not np.ndarray:
        raise TypeError(
            format_type_err_msg(
                "epi.util.gaussian_backward_mapping", "mu", mu, np.ndarray
            )
        )
    elif type(Sigma) is not np.ndarray:
        raise TypeError(
            format_type_err_msg(
                "epi.util.gaussian_backward_mapping", "Sigma", Sigma, np.ndarray
            )
        )

    mu = np_column_vec(mu)
    Sigma_shape = Sigma.shape
    if len(Sigma_shape) != 2:
        raise ValueError("Sigma must be 2D matrix, shape ", Sigma_shape, ".")
    if Sigma_shape[0] != Sigma_shape[1]:
        raise ValueError("Sigma must be square matrix, shape ", Sigma_shape, ".")
    if not np.allclose(Sigma, Sigma.T, atol=1e-10):
        raise ValueError("Sigma must be symmetric. shape.")
    if Sigma_shape[1] != mu.shape[0]:
        raise ValueError("mu and Sigma must have same dimensionality.")

    D = mu.shape[0]
    Sigma_inv = np.linalg.inv(Sigma)
    x = np.dot(Sigma_inv, mu)
    y = np.reshape(-0.5 * Sigma_inv, (D ** 2))
    eta = np.concatenate((x[:, 0], y), axis=0)
    return eta


def np_column_vec(x):
    """ Takes numpy vector and orients it as a n x 1 column vec. 

    :param x: Vector of length n
    :type x: np.ndarray
    :return: n x 1 numpy column vector
    :rtype: np.ndarray
    """
    if type(x) is not np.ndarray:
        raise (
            TypeError(format_type_err_msg("epi.util.np_column_vec", "x", x, np.ndarray))
        )
    x_shape = x.shape
    if len(x_shape) == 1:
        x = np.expand_dims(x, 1)
    elif len(x_shape) == 2:
        if x_shape[1] != 1:
            if x_shape[0] > 1:
                raise ValueError("x is matrix.")
            else:
                x = x.T
    elif len(x_shape) > 2:
        raise ValueError("x dimensions > 2.")
    return x


def array_str(a):
    """Returns a compressed string from a 1-D numpy array.

    :param a: A 1-D numpy array.
    :type a: str
    :return: A string compressed via scientific notation and repeated elements.
    :rtype: str
    """
    if type(a) is not np.ndarray:
        raise TypeError(format_type_err_msg("epi.util.array_str", "a", a, np.ndarray))

    if len(a.shape) > 1:
        raise ValueError("epi.util.array_str takes 1-D arrays not %d." % len(a.shape))

    def repeats_str(num, mult):
        if mult == 1:
            return "%.2E" % num
        else:
            return "%dx%.2E" % (mult, num)

    d = a.shape[0]
    if d == 1:
        nums = [a[0]]
        mults = [1]
    else:
        mults = []
        nums = []
        prev_num = a[0]
        mult = 1
        for i in range(1, d):
            if a[i] == prev_num:
                mult += 1
            else:
                nums.append(prev_num)
                prev_num = a[i]
                mults.append(mult)
                mult = 1

            if i == d - 1:
                nums.append(prev_num)
                mults.append(mult)

    array_str = repeats_str(nums[0], mults[0])
    for i in range(1, len(nums)):
        array_str += "_" + repeats_str(nums[i], mults[i])

    return array_str

def aug_lag_vars(z, log_q_z, eps, mu, N):
    """Calculate augmented lagrangian variables requiring gradient tape.

    :math:`H(\\theta) = \\mathbb{E}_{z \\sim q_\\theta}[-\\log(q_\\theta(z)]`

    :math:`R(\\theta) = \\mathbb{E}_{z \\sim q_\\theta, x \\sim p(x \mid z)}[T(x) - \\mu]`

    :math:`R_1(\\theta) = \\mathbb{E}_{z_1 \\sim q_\\theta, x \\sim p(x \mid z_1)}[T(x) - \\mu]`

    :math:`R_2(\\theta) = \\mathbb{E}_{z_2 \\sim q_\\theta, x \\sim p(x \mid z_2)}[T(x) - \\mu]`

    where :math:`\\theta` are params and :math:`z_1`, :math:`z_2` are the two 
    halves of the batch samples.

    :param z: Parameter samples.
    :type z: tf.Tensor
    :param log_q_z: Parameter sample log density.
    :type log_q_z: tf.Tensor
    :param eps: Emergent property statistics function.
    :type eps: function
    :param mu: Mean parameter of the emergent property.
    :type mu: np.ndarray
    :param N: Number of batch samples.
    :type N: int
    :return: :math:`H(\\theta)`, :math:`R(\\theta)`, list :math:`R_1(\\theta)` by dimension, and :math:`R_2(\\theta)`.
    :rtype: list

    """
    H = -tf.reduce_mean(log_q_z)
    T_x = eps(z)
    clip_lb = -1e10 * tf.ones_like(T_x, dtype=tf.float32)
    clip_ub = 1e10 * tf.ones_like(T_x, dtype=tf.float32)
    T_x = tf.clip_by_value(T_x, clip_lb, clip_ub)
    R = tf.reduce_mean(T_x, axis=0) - mu
    R1s = tf.unstack(tf.reduce_mean(T_x[: N // 2, :], 0) - mu, axis=0)
    R2 = tf.reduce_mean(T_x[N // 2 :, :], 0) - mu

    return H, R, R1s, R2


def unbiased_aug_grad(R1s, R2, params, tape):
    """Unbiased gradient of the l-2 norm of stochastic constraint violations.

    :math:`R_1(\\theta) = \\mathbb{E}_{z_1 \\sim q_\\theta, x \\sim p(x \mid z_1)}[T(x) - \\mu]`

    :math:`R_2(\\theta) = \\mathbb{E}_{z_2 \\sim q_\\theta, x \\sim p(x \mid z_2)}[T(x) - \\mu]`

    where :math:`\\theta` are params and :math:`z_1`, :math:`z_2` are the two 
    halves of the batch samples.

    The augmented gradient is computed as

    :math:`\\nabla_\\theta ||R(\\theta)||^2 = 2 \\nabla_\\theta R_1(\\theta) \\cdot R_2(\\theta)`

    :param R1s: Mean constraint violation over first half of samples.
    :type R1s: list
    :param R2: Mean constraint violation over the second half of samples.
    :type R2: tf.Tensor
    :param params: Trainable variables of :math:`q_\\theta`
    :type params: list
    :param tape: Persistent gradient tape watching params.
    :type tape: tf.GradientTape
    :return: Unbiased gradient of augmented term.
    :rtype: list
    """

    m = len(R1s)
    R1_grads = tape.gradient(R1s[0], params)
    jacR1 = [[g] for g in R1_grads]
    for i in range(1, m):
        R1_gradi = tape.gradient(R1s[i], params)
        for j, g in enumerate(R1_gradi):
            jacR1[j].append(g)

    jacR1 = [tf.stack(grad_list, axis=-1) for grad_list in jacR1]
    # We don't multiply by 2, since its cancels with the denominator
    # of the leading c/2 factor in the cost function.
    return [tf.linalg.matvec(jacR1i, R2) for jacR1i in jacR1]


class AugLagHPs:
    """Augmented Lagrangian optimization hyperparamters.

    :param N: Batch size, defaults to 1000.
    :type N: int, optional
    :param lr: Learning rate, defaults to 1e-3.
    :type lr: float, optional
    :param c0: L-2 norm on R coefficient, defaults to 1.0.
    :type c0: float, optional
    :param gamma: Epoch reduction factor for epoch, defaults to 1/4.
    :type gamma: float, optional
    :param beta: L-2 norm magnitude increase factor.
    :type beta: float, optional
    """

    def __init__(self, N=1000, lr=1e-3, c0=1.0, gamma=0.25, beta=4.0):
        self._set_N(N)
        self._set_lr(lr)
        self._set_c0(c0)
        self._set_gamma(gamma)
        self._set_beta(beta)

    def _set_N(self, N):
        if type(N) is not int:
            raise TypeError(format_type_err_msg(self, "N", N, int))
        elif N < 2:
            raise ValueError("N %d must be greater than 1." % N)
        self.N = N

    def _set_lr(self, lr):
        if type(lr) not in [float, np.float32, np.float64]:
            raise TypeError(format_type_err_msg(self, "lr", lr, float))
        elif lr < 0.0:
            raise ValueError("lr %.2E must be greater than 0." % lr)
        self.lr = lr

    def _set_c0(self, c0):
        if type(c0) not in [float, np.float32, np.float64]:
            raise TypeError(format_type_err_msg(self, "c0", c0, float))
        elif c0 < 0.0:
            raise ValueError("c0 %.2E must be greater than 0." % c0)
        self.c0 = c0

    def _set_gamma(self, gamma):
        if type(gamma) not in [float, np.float32, np.float64]:
            raise TypeError(format_type_err_msg(self, "gamma", gamma, float))
        elif gamma < 0.0:
            raise ValueError("gamma %.2E must be greater than 0." % gamma)
        self.gamma = gamma

    def _set_beta(self, beta):
        if type(beta) not in [float, np.float32, np.float64]:
            raise TypeError(format_type_err_msg(self, "beta", beta, float))
        elif beta < 0.0:
            raise ValueError("beta %.2E must be greater than 0." % beta)
        self.beta = beta

    def to_string(self,):
        """String for filename involving hyperparameter setting.

        :returns: Hyperparameters as a string.
        :rtype: str
        """
        return "N%d_lr%.2E_c0=%.2E_gamma%.2E_beta%.2E" % (
            self.N,
            self.lr,
            self.c0,
            self.gamma,
            self.beta,
        )


def check_bound_param(bounds, param_name):
    if type(bounds) not in [list, tuple]:
        raise TypeError(
            format_type_err_msg("sample_aug_lag_hps", param_name, bounds, list)
        )
    if len(bounds) != 2:
        raise ValueError("Bound should be length 2.")
    if bounds[1] < bounds[0]:
        raise ValueError("Bounds are not ordered correctly: bounds[1] < bounds[0].")
    return None


def sample_aug_lag_hps(
    n,
    N_bounds=[200, 1000],
    lr_bounds=[1e-4, 1e-2],
    c0_bounds=[1e-3, 1e3],
    gamma_bounds=[0.1, 0.5],
):
    """Samples augmented Lagrangian parameters from uniform distribution.

    :param N_bounds: Bounds on batch size.
    """

    check_bound_param(N_bounds, "N_bounds")
    check_bound_param(lr_bounds, "lr_bounds")
    check_bound_param(c0_bounds, "c0_bounds")
    check_bound_param(gamma_bounds, "gamma_bounds")

    if N_bounds[0] < 10:
        raise ValueError(
            "Batch size should be greater than 10. Lower bound set to %d." % N_bounds[0]
        )
    if lr_bounds[0] < 0.0:
        raise ValueError(
            "Learning rate must be positive. Lower bound set to %.2E." % lr_bounds[0]
        )
    if c0_bounds[0] <= 0.0:
        raise ValueError(
            "Initial augmented Lagrangian coefficient c0 must be positive. Lower bound set to %.2E."
            % c0_bounds[0]
        )
    if gamma_bounds[0] < 0.0 or gamma_bounds[1] > 1.0:
        raise ValueError("gamma parameter must be from 0 to 1.")

    aug_lag_hps = []
    for i in range(n):
        N = np.random.randint(N_bounds[0], N_bounds[1])
        lr = np.exp(np.random.uniform(np.log(lr_bounds[0]), np.log(lr_bounds[1])))
        c0 = np.exp(np.random.uniform(np.log(c0_bounds[0]), np.log(c0_bounds[1])))
        gamma = np.random.uniform(gamma_bounds[0], gamma_bounds[1])
        beta = 1.0 / gamma
        aug_lag_hp_i = AugLagHPs(N, lr, c0, gamma, beta)
        aug_lag_hps.append(aug_lag_hp_i)

    if n == 1:
        return aug_lag_hps[0]
    else:
        return aug_lag_hps


def plot_square_mat(
    ax,
    A,
    c="k",
    lw=4,
    fontsize=12,
    bfrac=0.05,
    title=None,
    xlims=None,
    ylims=None,
    text_c="k",
):
    buf = 0.3
    if xlims is None:
        ax.set_xlim([-0.05, 1 + buf])
    else:
        ax.set_xlim(xlims)
    if ylims is None:
        ax.set_ylim([-0.05, 1 + buf])
    else:
        ax.set_ylim(ylims)

    D = A.shape[0]
    if len(A) != 2:
        raise ValueError("A is not 2D.")
    if A.shape[1] != D:
        raise ValueError("A is not square")
    xs = np.linspace(1.0 / (2.0 * D), 1 - 1.0 / (2.0 * D), D)
    ys = np.linspace(1.0 - 1.0 / (2.0 * D), 1.0 / (2.0 * D), D) - 0.02

    shift_x1 = -0.18 / D
    shift_x2 = -0.35 / D
    shift_y1 = +0.2 / D
    shift_y2 = -0.2 / D
    texts = []
    for i in range(D):
        for j in range(D):
            ax.text(
                xs[j] + shift_x1,
                ys[i] + shift_y1,
                r"$a_{%d%d}$" % (i + 1, j + 1),
                fontsize=(fontsize - 2),
            )
            texts.append(
                ax.text(
                    xs[j] + shift_x2,
                    ys[i] + shift_y2,
                    "%.2f" % A[i, j],
                    fontsize=fontsize,
                    color=text_c,
                    weight="bold",
                )
            )

    ax.plot([0, 0], [0, 1], "k", c=c, lw=lw)
    ax.plot([0, bfrac], [0, 0], "k", c=c, lw=lw)
    ax.plot([0, bfrac], [1, 1], "k", c=c, lw=lw)
    ax.plot([1, 1], [0, 1], "k", c=c, lw=lw)
    ax.plot([1.0 - bfrac, 1], [0, 0], "k", c=c, lw=lw)
    ax.plot([1.0 - bfrac, 1], [1, 1], "k", c=c, lw=lw)
    if title is not None:
        ax.text(0.27, 1.1, title, fontsize=(fontsize - 4))
    ax.axis("off")
    return texts

def old_pairplot(
    Z,
    dims,
    labels,
    z1=None,
    z2=None,
    origin=False,
    xlims=None,
    ylims=None,
    clims=None,
    ticks=None,
    c=None,
    c_label=None,
    c_traj=None,
    cmap=None,
    ss=False,
    fontsize=12,
    figsize=(12, 12),
    outlier_stds=10,
    pfname="images/temp.png",
):
    M = Z.shape[0]
    num_dims = len(dims)
    rand_order = np.random.permutation(M)
    Z = Z[rand_order, :]
    if c is not None:
        c = c[rand_order]
        if clims is not None:
            all_inds = np.arange(c.shape[0])
            below_inds = all_inds[c < clims[0]]
            over_inds = all_inds[c > clims[1]]
            plot_inds = all_inds[
                np.logical_and(clims[0] <= c, c <= clims[1])
            ]
        else:
            plot_inds, below_inds, over_inds = filter_outliers(c, outlier_stds)
            clims = [None, None]

    fig, axs = plt.subplots(num_dims - 1, num_dims - 1, figsize=figsize)
    for i in range(num_dims - 1):
        dim_i = dims[i]
        for j in range(1, num_dims):
            if num_dims == 2:
                ax = plt.gca()
            else:
                ax = axs[i, j - 1]
            if j > i:
                dim_j = dims[j]
                if (xlims is not None) and (ylims is not None) and origin:
                    ax.plot(xlims, [0, 0], c=0.5 * np.ones(3), linestyle="--")
                    ax.plot([0, 0], ylims, c=0.5 * np.ones(3), linestyle="--")
                if ss:
                    ax.plot(
                        np.reshape(Z[:, dim_j].T, (M // 2, 2)),
                        np.reshape(Z[:, dim_i].T, (M // 2, 2)),
                        "k",
                        lw=0.2,
                    )
                if c is not None:
                    ax.scatter(
                        Z[below_inds, dim_j],
                        Z[below_inds, dim_i],
                        c="k",
                        edgecolors="k",
                        linewidths=0.25,
                    )
                    ax.scatter(
                        Z[over_inds, dim_j],
                        Z[over_inds, dim_i],
                        c="w",
                        edgecolors="k",
                        linewidths=0.25,
                    )
                    h = ax.scatter(
                        Z[plot_inds, dim_j],
                        Z[plot_inds, dim_i],
                        c=c[plot_inds],
                        cmap=cmap,
                        vmin=clims[0],
                        vmax=clims[1],
                        edgecolors="k",
                        linewidths=0.25,
                    )
                else:
                    h = ax.scatter(
                        Z[:, dim_j], Z[:, dim_i], c='k', edgecolors="k", linewidths=0.25,
                    )
                if (z1 is not None):
                    if (z1.shape[0] == z2.shape[0]):
                        for _i in range(len(z1)):
                            fac = 0.9 # percentage of length where arrow head should start
                            xy_st = [z1[_i,j], z1[_i,i]]
                            xy_mid = [((1-fac)*z1[_i,j])+(fac*z2[_i,j]), 
                                      ((1-fac)*z1[_i,i])+(fac*z2[_i,i])]
                            xy_end = [z2[_i,j], z2[_i,i]]
                            norm = np.linalg.norm(np.array(xy_end) - np.array(xy_st))
                            if c_traj is not None:
                                _c = c_traj[_i]
                            else:
                                _c = 'k'
                            if xlims is not None:
                                lims = xlims
                            else:
                                lims = [np.min(Z), np.max(Z)]
                            if norm > 1.:
                                ax.plot(
                                    [xy_st[0], xy_mid[0]], 
                                    [xy_st[1], xy_mid[1]], 
                                    '-',
                                    lw=5,
                                    c=_c,
                                )
                                ax.annotate(
                                    "", 
                                    xy=xy_end,
                                    xytext=xy_mid, 
                                    arrowprops=dict(headwidth=norm*10, lw=2, color=_c)
                                )
                    else:
                        print('Error: z1 and z2 had different lengths.')
                        pass
                if i + 1 == j:
                    ax.set_xlabel(labels[j], fontsize=fontsize)
                    ax.set_ylabel(labels[i], fontsize=fontsize)
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                if ticks is not None:
                    ax.set_xticks(ticks, fontsize=fontsize)
                    ax.set_yticks(ticks, fontsize=fontsize)
                if xlims is not None:
                    ax.set_xlim(xlims)
                if ylims is not None:
                    ax.set_ylim(ylims)
            else:
                ax.axis("off")

    if c is not None:
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.04, 0.7])
        clb = fig.colorbar(h, cax=cbar_ax)
        a = (1.01 / (num_dims - 1)) / (0.9 / (num_dims - 1))
        b = (num_dims - 1) * 1.15
        plt.text(a, b, c_label, {"fontsize": fontsize}, transform=ax.transAxes)
    # plt.savefig(pfname)
    return fig, axs

def filter_outliers(c, num_stds=4):
    max_stat = 10e5
    _c = c[np.logical_and(c < max_stat, c > -max_stat)]
    c_mean = np.mean(_c)
    c_std = np.std(_c)
    all_inds = np.arange(c.shape[0])
    below_inds = all_inds[c < c_mean - num_stds * c_std]
    over_inds = all_inds[c > c_mean + num_stds * c_std]
    plot_inds = all_inds[
        np.logical_and(c_mean - num_stds * c_std <= c, c <= c_mean + num_stds * c_std)
    ]
    return plot_inds, below_inds, over_inds




def pairplot(
    Z,
    dims,
    labels,
    lb = None,
    ub = None,
    clims=None,
    ticks=None,
    c=None,
    c_label=None,
    cmap=None,
    s=10,
    starred=None,
    c_starred = None,
    fontsize=12,
    figsize=(12, 12),
    outlier_stds=10,
    ticksize=None,
    pfname="images/temp.png",
):
    M = Z.shape[0]
    num_dims = len(dims)
    rand_order = np.random.permutation(M)
    Z = Z[rand_order, :]
    if c is not None:
        c = c[rand_order]
        if clims is not None:
            all_inds = np.arange(c.shape[0])
            below_inds = all_inds[c < clims[0]]
            over_inds = all_inds[c > clims[1]]
            plot_inds = all_inds[
                np.logical_and(clims[0] <= c, c <= clims[1])
            ]
        else:
            plot_inds, below_inds, over_inds = filter_outliers(c, outlier_stds)
            clims = [None, None]
    if ticksize is None:
        ticksize = fontsize-4

    fig, axs = plt.subplots(num_dims - 1, num_dims - 1, figsize=figsize)
    for i in range(num_dims - 1):
        dim_i = dims[i]
        for j in range(1, num_dims):
            if num_dims == 2:
                ax = plt.gca()
            else:
                ax = axs[i, j - 1]
            if j > i:
                dim_j = dims[j]
                if c is not None:
                    ax.scatter(
                        Z[below_inds, dim_j],
                        Z[below_inds, dim_i],
                        c="k",
                        s=s,
                        edgecolors="k",
                        linewidths=0.25,
                    )
                    ax.scatter(
                        Z[over_inds, dim_j],
                        Z[over_inds, dim_i],
                        c="w",
                        s=s,
                        edgecolors="k",
                        linewidths=0.25,
                    )
                    h = ax.scatter(
                        Z[plot_inds, dim_j],
                        Z[plot_inds, dim_i],
                        c=c[plot_inds],
                        cmap=cmap,
                        s=s,
                        vmin=clims[0],
                        vmax=clims[1],
                        edgecolors="k",
                        linewidths=0.25,
                    )
                else:
                    h = ax.scatter(
                        Z[:, dim_j], Z[:, dim_i], c='k', s=s, edgecolors="k", linewidths=0.25,
                    )
                if starred is not None:
                    if c_starred is None:
                        ax.scatter(
                            starred[:, dim_j], starred[:, dim_i], s=400, c='k', 
                            marker='*', edgecolors="k", linewidths=0.5,
                        )
                    else:
                        ax.scatter(
                            starred[:, dim_j], starred[:, dim_i], s=400, c=c_starred, 
                            marker='*', edgecolors="k", linewidths=0.5,
                        )

                if i + 1 == j:
                    ax.set_xlabel(labels[j], fontsize=fontsize)
                    ax.set_ylabel(labels[i], fontsize=fontsize)
                    plt.setp(ax.get_xticklabels(), fontsize=ticksize)
                    plt.setp(ax.get_yticklabels(), fontsize=ticksize)
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                if ticks is not None:
                    ax.set_xticks(ticks, fontsize=fontsize)
                    ax.set_yticks(ticks, fontsize=fontsize)
                if lb is not None and ub is not None:
                    ax.set_xlim(lb[dim_j], ub[dim_j])
                if lb is not None and ub is not None:
                    ax.set_ylim(lb[dim_i], ub[dim_i])
            else:
                ax.axis("off")

    if c is not None:
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.04, 0.7])
        clb = fig.colorbar(h, cax=cbar_ax)
        a = (1.01 / (num_dims - 1)) / (0.9 / (num_dims - 1))
        b = (num_dims - 1) * 1.15
        plt.text(a, b, c_label, {"fontsize": fontsize}, transform=ax.transAxes)
        clb.ax.tick_params(labelsize=ticksize)
    # plt.savefig(pfname)
    return fig, axs

def filter_outliers(c, num_stds=4):
    max_stat = 10e5
    _c = c[np.logical_and(c < max_stat, c > -max_stat)]
    c_mean = np.mean(_c)
    c_std = np.std(_c)
    all_inds = np.arange(c.shape[0])
    below_inds = all_inds[c < c_mean - num_stds * c_std]
    over_inds = all_inds[c > c_mean + num_stds * c_std]
    plot_inds = all_inds[
        np.logical_and(c_mean - num_stds * c_std <= c, c <= c_mean + num_stds * c_std)
    ]
    return plot_inds, below_inds, over_inds

purple = '#4C0099'
def plot_T_x(T_x, T_x_sim, bins=30, xmin=None, xmax=None, 
             x_mean=None, x_std=None,
             xlabel=None, ylim=None, fontsize=14):
    if xmin is not None and xmax is not None:
        _range = (xmin, xmax)
    else:
        _range = (x_mean - 4*x_std, x_mean + 4*x_std)
    fig, ax = plt.subplots(1,1)
    ['ABC simulations', 'ABC posterior predictive']
    if T_x is None:
        ax.hist(T_x_sim, bins=bins, range=_range, color=purple, 
                alpha=0.5, label='ABC posterior predictive')
    else:
        n, bins, patches = ax.hist(T_x, bins=bins, color='k', range=_range, 
                alpha=0.5, label='ABC simulations')
        ax.hist(T_x_sim, bins=bins, color=purple, alpha=0.5, label='ABC posterior predictive')
    if ylim is not None:
        ax.set_ylim(ylim)
    ylim = ax.get_ylim()
    ax.plot([x_mean, x_mean], ylim, 'k--')
    ax.plot([x_mean+2*x_std, x_mean+2*x_std], ylim, '--', c=[0.5, 0.5, 0.5])
    ax.plot([x_mean-2*x_std, x_mean-2*x_std], ylim, '--', c=[0.5, 0.5, 0.5])

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    xticks = [_range[0], x_mean-2*x_std, x_mean, x_mean+2*x_std, _range[1]]
    xticks = np.around(xticks, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=(fontsize-4))
    plt.setp(ax.get_yticklabels(), fontsize=(fontsize-4))
    ax.set_ylabel('count', fontsize=fontsize)

    return ax
