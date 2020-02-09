import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns

from epi.models import Model, Parameter

# Mac OS jupyter kernel dies without
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# bounds = [-10., 10.]
a11 = Parameter("a11")
a12 = Parameter("a12")
a21 = Parameter("a21")
a22 = Parameter("a22")

params = [a11, a12, a21, a22]
M = Model("lds", params)

# Emergent property statistics
from epi.example_eps import linear2D_freq

m = 4  # number of emergent propety statistics
M.set_eps(linear2D_freq, m)

# Emergent property values
mu = np.array([0.0, 0.25 ** 2, 2 * np.pi, (0.2 * np.pi) ** 2])

init_params = {"loc": 0.0, "scale": 3.0}
q_theta, opt_data, save_path = M.epi(
    mu,
    arch_type="autoregressive",
    num_stages=1,
    num_layers=2,
    num_units=15,
    post_affine=True,
    init_params=init_params,
    K=1,
    N=200,
    num_iters=200,
    lr=1e-4,
    c0=1e-3,
    verbose=True,
    log_rate=25,
    save_movie_data=True,
)

M.epi_opt_movie(save_path)
