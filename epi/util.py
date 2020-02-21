""" General util functions for EPI. """

import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KernelDensity
from epi.error_formatters import format_type_err_msg


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


def init_path(arch_string, init_type, init_params):
    """Deduces initialization file path from initialization type and parameters.

    :param arch_string: Architecture string of normalizing flow.
    :type arch_string: str
    :param init_type: Initialization type \in ['iso_gauss']
    :type init_type: str
    :param init_param: init_type dependent parameters for initialization (more deets)
    :type dict: 

    :return: Initialization save path.
    :rtype: str
    """
    if type(arch_string) is not str:
        raise TypeError(
            format_type_err_msg("epi.util.init_path", "arch_string", arch_string, str)
        )
    if type(init_type) is not str:
        raise TypeError(
            format_type_err_msg("epi.util.init_path", "init_type", init_type, str)
        )

    path = "./data/" + arch_string + "/"

    if init_type == "iso_gauss":
        if "loc" in init_params:
            loc = init_params["loc"]
        else:
            raise ValueError("'loc' field not in init_param for %s." % init_type)
        if "scale" in init_params:
            scale = init_params["scale"]
        else:
            raise ValueError("'scale' field not in init_param for %s." % init_type)
        path += init_type + "_loc=%.2E_scale=%.2E/" % (loc, scale)
    elif init_type == "gaussian":
        if "mu" in init_params:
            mu = np_column_vec(init_params["mu"])[:, 0]
        else:
            raise ValueError("'mu' field not in init_param for %s." % init_type)
        if "Sigma" in init_params:
            Sigma = init_params["Sigma"]
        else:
            raise ValueError("'Sigma' field not in init_param for %s." % init_type)
        D = mu.shape[0]
        mu_str = array_str(mu)
        Sigma_str = array_str(Sigma[np.triu_indices(D, 0)])
        path += init_type + "_mu=%s_Sigma=%s/" % (mu_str, Sigma_str)

    if not os.path.exists(path):
        os.makedirs(path)

    return path


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

    if (n==1):
        return aug_lag_hps[0]
    else:
        return aug_lag_hps

def plot_square_mat(ax, A, c='k', lw=4, fontsize=12, bfrac=0.05, title=None):
    buf = .15
    ax.set_xlim([-.05, 1+buf])
    ax.set_ylim([-.05, 1+buf])
    D = A.shape[0]
    if (len(A) != 2):
        raise ValueError("A is not 2D.")
    if (A.shape[1] != D):
        raise ValueError("A is not square")
    xs = np.linspace(1./(2.*D), 1 - 1./(2.*D), D)
    ys = np.linspace(1.- 1./(2.*D), 1./(2.*D), D)-.02

    shift_x1 = -.18/D
    shift_x2 = -.35/D
    shift_y1 = +.2/D
    shift_y2 = -.2/D
    texts = []
    for i in range(D):
        for j in range(D):
            ax.text(xs[j]+shift_x1, ys[i]+shift_y1, r"$a_{%d%d}$" % (i+1, j+1), fontsize=(fontsize-2))
            texts.append(ax.text(xs[j]+shift_x2, ys[i]+shift_y2, "%.2f" % A[i,j], fontsize=fontsize))

    ax.plot([0,0], [0,1], 'k', c=c, lw=lw)
    ax.plot([0,bfrac], [0,0], 'k', c=c, lw=lw)
    ax.plot([0,bfrac], [1,1], 'k', c=c, lw=lw)
    ax.plot([1,1], [0,1], 'k', c=c, lw=lw)
    ax.plot([1.-bfrac, 1], [0,0], 'k', c=c, lw=lw)
    ax.plot([1.-bfrac, 1], [1,1], 'k', c=c, lw=lw)
    if (title is not None):
        ax.set_title(title, fontsize=(fontsize-5), pad=-26.)
    ax.axis('off')
    return texts


def lds_2D_movie(path, parameters, z_labels=None):
    """Generate video of EPI for 2D lds.

    :param path: Path to folder with optimization data.
    :type param: str
    """
    fontsize=22
    if (z_labels is None):
        z_labels = []
        z_labels = [parameter.name for parameter in parameters]
    D = len(z_labels)
    palette = sns.color_palette()

    z_filename = path + "movie_data.npz"
    opt_data_filename = path + "opt_data.csv"
    # Load zs for optimization.
    if os.path.exists(z_filename):
        z_file = np.load(z_filename)
    else:
        raise IOError("File %s does not exist." % z_filename)
    if os.path.exists(opt_data_filename):
        opt_data_df = pd.read_csv(opt_data_filename)
    else:
        raise IOError("File %s does not exist." % opt_data_filename)

    zs = z_file["zs"]
    log_q_zs = z_file["log_q_zs"]
    iters = z_file["iterations"]

    N_frames = len(iters)
    Hs = opt_data_df["H"]
    if (len(Hs) < N_frames) or (not np.isclose(iters, opt_data_df["iteration"][:N_frames]).all()):
        raise IOError("opt_data.csv incompatible with movie_data.npz.")
    R_keys = []
    for key in opt_data_df.columns:
        if "R" in key:
            R_keys.append(key)
    m = len(R_keys)
    R = opt_data_df[R_keys].to_numpy()

    _iters = [iters[0]]
    _Hs = [Hs[0]]
    z = zs[0]
    log_q_z = log_q_zs[0]
    ylab_x = -0.075
    ylab_y = 0.6

    iter_rows = 3
    # Entropy lines
    fig, axs = plt.subplots(4, 8, figsize=(16, 8))
    H_ax = plt.subplot(4, 2, 1)
    H_ax.set_xlim(0, iters[-1])
    min_H, max_H = np.min(Hs), np.max(Hs)
    H_ax.set_ylim(min_H, max_H)
    H_line, = H_ax.plot(_iters, _Hs, c=palette[0])
    H_ax.set_ylabel(r"$H(q_\theta)$", rotation="horizontal")
    H_ax.yaxis.set_label_coords(ylab_x-.05, ylab_y)

    # Constraint lines
    R_ax = plt.subplot(4, 2, 3) 

    R_ax.set_xlim(0, iters[-1])
    min_R, max_R = np.min(R), np.max(R)

    R_ax.set_ylim(min_R, max_R)
    R_ax.set_ylabel(r"$R(q_\theta)$", rotation="horizontal")
    R_ax.yaxis.set_label_coords(ylab_x-.05, ylab_y)
    R_ax.set_xlabel("iterations")

    R_lines = []
    _Rs = []
    for i in range(m):
        _Rs.append([R[0, i]])
        R_line, = R_ax.plot(_iters, _Rs[i], label=R_keys[i], c=palette[i + 1])
        R_lines.append(R_line)
    R_ax.legend()

    lines = [H_line] + R_lines

    # Get axis limits
    ax_mins = []
    ax_maxs = []
    for i in range(D):
        bounds = parameters[i].bounds
        if np.isneginf(bounds[0]):
            ax_mins.append(np.min(zs[:, :, i]))
        else:
            ax_mins.append(bounds[0])

        if np.isposinf(bounds[1]):
            ax_maxs.append(np.max(zs[:, :, i]))
        else:
            ax_maxs.append(bounds[1])

    # Plot the matrices
    def get_lds_2D_modes(z, log_q_z):
        M = log_q_z.shape[0]
        mode1 = np.logical_and(z[:,1] > 0., z[:,2] < 0)
        mode1_inds = np.arange(M)[mode1]
        mode2 = np.logical_and(z[:,1] < 0., z[:,2] > 0)
        mode2_inds = np.arange(M)[mode2]
        mode1_ind = mode1_inds[np.argmax(log_q_z[mode1])]
        mode2_ind = mode2_inds[np.argmax(log_q_z[mode2])]
        return np.reshape(z[mode1_ind], (2,2)), np.reshape(z[mode2_ind], (2,2))

    mode1, mode2 = get_lds_2D_modes(z, log_q_z)
    lw=5
    gray = 0.4*np.ones(3)
    bfrac=0.05
    mat_ax = plt.subplot(2, 4, 5)
    texts = plot_square_mat(mat_ax, mode1, c=gray, lw=lw, fontsize=24, bfrac=bfrac, title="mode 1")
    mat_ax = plt.subplot(2, 4, 6)
    texts += plot_square_mat(mat_ax, mode2, c=gray, lw=lw, fontsize=24, bfrac=bfrac, title="mode 2")


    # Collect scatters
    cmap = plt.get_cmap("viridis")
    scats = []
    for i in range(D - 1):
        for j in range(i + 1, D):
            ax = axs[i][j+4]
            scats.append(ax.scatter(z[:, j], z[:, i], c=log_q_z, cmap=cmap))
            ax.set_xlim(ax_mins[j], ax_maxs[j])
            ax.set_ylim(ax_mins[i], ax_maxs[i])

    kdes = []
    conts = []
    kde_scale_fac = 0.1
    num_grid = 100
    for i in range(1, D):
        ax_len_i = ax_maxs[i] - ax_mins[i]
        grid_xs = np.linspace(ax_mins[i], ax_maxs[i], num_grid)
        for j in range(i):
            ax_len_j = ax_maxs[i] - ax_mins[i]
            kde = KernelDensity(
                kernel="gaussian",
                bandwidth=kde_scale_fac * (ax_len_i + ax_len_j) / 2.0,
            )
            _z = z[:, [i, j]]
            kde.fit(_z, log_q_z)
            grid_ys = np.linspace(ax_mins[j], ax_maxs[j], num_grid)
            z_grid = np.meshgrid(grid_xs, grid_ys)
            z_grid_mat = np.stack([np.reshape(z_grid[0], (num_grid**2)),
                               np.reshape(z_grid[1], (num_grid**2))],
                              axis=1)
            scores_ij = kde.score_samples(z_grid_mat)
            scores_ij = np.reshape(scores_ij, (num_grid, num_grid))
            ax = axs[i][j+4]
            levels = np.linspace(np.min(scores_ij), np.max(scores_ij), 20)
            cont = ax.contourf(z_grid[0], z_grid[1], scores_ij, levels=levels)
            conts.append(cont)
            ax.set_xlim(ax_mins[i], ax_maxs[i])
            ax.set_ylim(ax_mins[j], ax_maxs[j])
            kdes.append(kde)

    for i in range(D):
        axs[i][4].set_ylabel(
            z_labels[i], rotation="horizontal", fontsize=fontsize
        )
        axs[i][4].yaxis.set_label_coords(D * ylab_x, ylab_y)
        axs[3][i+4].set_xlabel(z_labels[i], fontsize=fontsize)
        axs[i][i+4].set_xlim(ax_mins[i], ax_maxs[i])
        axs[i][i+4].set_ylim(ax_mins[i], ax_maxs[i])

    for i in range(D):
        for j in range(1, D):
            axs[i, j+4].set_yticklabels([])
    for i in range(D - 1):
        for j in range(D):
            axs[i, j+4].set_xticklabels([])

    def update(frame):
        _iters.append(iters[frame])
        _Hs.append(Hs[frame])
        for i in range(m):
            _Rs[i].append(R[frame, i])
        z = zs[frame]
        log_q_z = log_q_zs[frame]
        cvals = log_q_z - np.min(log_q_z)
        cvals = cvals / np.max(cvals)

        # Update entropy plot
        lines[0].set_data(_iters, _Hs)
        for i in range(m):
            lines[i + 1].set_data(_iters, _Rs[i])

        # Update modes.
        mode1, mode2 = get_lds_2D_modes(z, log_q_z)
        ind = 0
        for i in range(2):
            for j in range(2):
                texts[ind].set_text('%.1f' % mode1[i,j])
                texts[ind+4].set_text('%.1f' % mode2[i,j])
                ind += 1

        # Update scatters
        _ind = 0
        for i in range(D - 1):
            for j in range(i + 1, D):
                scats[_ind].set_offsets(np.stack((z[:, j], z[:, i]), axis=1))
                scats[_ind].set_color(cmap(cvals))
                _ind += 1

        while conts:
            cont = conts.pop(0)
            for coll in cont.collections:
                coll.remove()

        _ind = 0
        for i in range(1, D):
            grid_x = np.linspace(ax_mins[i], ax_maxs[i], num_grid)
            for j in range(i):
                grid_ys = np.linspace(ax_mins[j], ax_maxs[j], num_grid)
                kde = kdes[_ind]
                _z = z[:, [i, j]]
                kde.fit(_z, log_q_z)
                grid_ys = np.linspace(ax_mins[j], ax_maxs[j], num_grid)
                z_grid = np.meshgrid(grid_xs, grid_ys)
                z_grid_mat = np.stack([np.reshape(z_grid[0], (num_grid**2)),
                                   np.reshape(z_grid[1], (num_grid**2))],
                                  axis=1)
                scores_ij = kde.score_samples(z_grid_mat)
                scores_ij = np.reshape(scores_ij, (num_grid, num_grid))
                ax = axs[i][j+4]
                levels = np.linspace(np.min(scores_ij), np.max(scores_ij), 20)
                cont = ax.contourf(z_grid[0], z_grid[1], scores_ij, levels=levels)
                conts.append(cont)

        return lines + scats

    ani = animation.FuncAnimation(fig, update, frames=range(N_frames), blit=True)

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=30, metadata=dict(artist="Me"), bitrate=1800)

    ani.save(path + "epi_opt.mp4", writer=writer)
    return None
