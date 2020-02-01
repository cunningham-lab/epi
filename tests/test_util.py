""" Test util functions. """

import numpy as np
import tensorflow as tf
from epi.util import (
    gaussian_backward_mapping,
    np_column_vec,
    array_str,
    init_path,
    save_tf_model,
    load_tf_model,
)
import pytest
from pytest import raises
import os


def test_gaussian_backward_mapping():
    """ Test gaussian_backward_mapping. """
    eta = gaussian_backward_mapping(np.zeros(2), np.eye(2))
    eta_true = np.array([0.0, 0.0, -0.5, 0.0, 0.0, -0.5])
    assert np.equal(eta, eta_true).all()

    eta = gaussian_backward_mapping(np.zeros(2), 2.0 * np.eye(2))
    eta_true = np.array([0.0, 0.0, -0.25, 0.0, 0.0, -0.25])
    assert np.equal(eta, eta_true).all()

    eta = gaussian_backward_mapping(2.0 * np.ones(2), np.eye(2))
    eta_true = np.array([2.0, 2.0, -0.5, 0.0, 0.0, -0.5])
    assert np.equal(eta, eta_true).all()

    mu = np.array([-4.0, 1000, 0.0001])
    Sigma = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    eta_true = np.concatenate(
        (np.dot(Sigma_inv, mu), np.reshape(-0.5 * Sigma_inv, (9,)))
    )
    eta = gaussian_backward_mapping(mu, Sigma)
    assert np.equal(eta, eta_true).all()

    with raises(TypeError):
        gaussian_backward_mapping("foo", np.eye(2))
    with raises(TypeError):
        gaussian_backward_mapping(np.zeros(2), "foo")

    with raises(ValueError):
        gaussian_backward_mapping(np.zeros(2), np.random.normal(0.0, 1.0, (2, 2, 2)))
    with raises(ValueError):
        gaussian_backward_mapping(np.zeros(2), np.random.normal(0.0, 1.0, (2, 3)))
    with raises(ValueError):
        gaussian_backward_mapping(np.zeros(2), np.random.normal(0.0, 1.0, (2, 2)))
    with raises(ValueError):
        eta = gaussian_backward_mapping(np.zeros(2), np.eye(3))
    assert np.equal(eta, eta_true).all()

    return None


def test_np_column_vec():
    """ Test np_column_vec. """
    x = np.random.uniform(-100, 100, (20, 1))
    X = np.random.uniform(-100, 100, (20, 20))
    X3 = np.random.uniform(-100, 100, (10, 10, 10))
    assert np.equal(x, np_column_vec(x[:, 0])).all()
    assert np.equal(x, np_column_vec(x.T)).all()

    with raises(TypeError):
        np_column_vec("foo")

    with raises(ValueError):
        np_column_vec(X)

    with raises(ValueError):
        np_column_vec(X3)
    return None


def test_array_str():
    a = np.array([3.2, 15.2, -0.4, -9.2])
    a_str = "3.20E+00_1.52E+01_-4.00E-01_-9.20E+00"
    assert a_str == array_str(a)

    a = np.array([3.2, 3.2, 15.2, -0.4, -9.2])
    a_str = "2x3.20E+00_1.52E+01_-4.00E-01_-9.20E+00"
    assert a_str == array_str(a)

    a = np.array([3.2, 15.2, 15.2, 15.2, -0.4, -9.2])
    a_str = "3.20E+00_3x1.52E+01_-4.00E-01_-9.20E+00"
    assert a_str == array_str(a)

    a = np.array([3.2, 15.2, -0.4, -9.2, -9.2, -9.2, -9.2])
    a_str = "3.20E+00_1.52E+01_-4.00E-01_4x-9.20E+00"
    assert a_str == array_str(a)

    a = np.array([3.2, 3.2, 15.2, -0.4, -9.2, -9.2])
    a_str = "2x3.20E+00_1.52E+01_-4.00E-01_2x-9.20E+00"
    assert a_str == array_str(a)

    with raises(TypeError):
        array_str("foo")

    with raises(ValueError):
        array_str(np.random.normal(0.0, 1.0, (3, 3)))

    return None


def test_init_path():
    arch_string = "foo"
    init_type = "iso_gauss"
    init_param = {"loc": 0.0, "scale": 1.0}

    s = init_path(arch_string, init_type, init_param)
    s_true = "./data/foo/iso_gauss_loc=0.00E+00_scale=1.00E+00"
    assert s == s_true

    with raises(TypeError):
        init_path(1, init_type, init_param)

    with raises(TypeError):
        init_path(arch_string, 1, init_param)

    init_param = {"scale": 1.0}
    with raises(ValueError):
        init_path(arch_string, init_type, init_param)

    init_param = {"loc": 0.0}
    with raises(ValueError):
        init_path(arch_string, init_type, init_param)
    return None


@pytest.fixture
def tf_image_classifier1():
    """ Basic model from the tf 2.0 tutorial. """
    tf.keras.backend.clear_session()
    mnist = tf.keras.datasets.mnist

    (x_train, _), _ = mnist.load_data()
    x_train = x_train[:32] / 255.0
    x_train = x_train[..., tf.newaxis]

    tf.random.set_seed(1)
    np.random.seed(1)

    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu")
            self.flatten = tf.keras.layers.Flatten()
            self.d1 = tf.keras.layers.Dense(128, activation="relu")
            self.d2 = tf.keras.layers.Dense(10, activation="softmax")

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    model = MyModel()
    model(x_train)
    return model


@pytest.fixture
def tf_image_classifier2():
    """ Basic model from the tf 2.0 tutorial. """
    tf.keras.backend.clear_session()
    mnist = tf.keras.datasets.mnist

    (x_train, _), _ = mnist.load_data()
    x_train = x_train[:32] / 255.0
    x_train = x_train[..., tf.newaxis]

    tf.random.set_seed(2)
    np.random.seed(2)

    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu")
            self.flatten = tf.keras.layers.Flatten()
            self.d1 = tf.keras.layers.Dense(128, activation="relu")
            self.d2 = tf.keras.layers.Dense(10, activation="softmax")

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    model = MyModel()
    model(x_train)
    return model


def test_save_and_load_tf_model(tf_image_classifier1, tf_image_classifier2):
    def tf_var_equal(vars1, vars2):
        # Make sure some variables are different between the two models.
        num_vars_diff = 0
        for var1, var2 in zip(tfic1_vars, tfic2_vars):
            if not np.isclose(var1.numpy(), var2.numpy()).all():
                num_vars_diff += 1
        return num_vars_diff == 0

    tfic1_vars = tf_image_classifier1.trainable_variables
    tfic2_vars = tf_image_classifier2.trainable_variables
    num_vars = len(tfic1_vars)
    assert num_vars > 0
    assert num_vars == len(tfic2_vars)

    # Parameters of two models should be different.
    assert not tf_var_equal(tfic1_vars, tfic2_vars)

    model_dir = "./test_models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model parameters with pickle.
    path1 = model_dir + "tmp_model1"
    save_tf_model(path1, tfic1_vars)
    path2 = model_dir + "tmp_model2"
    save_tf_model(path2, tfic2_vars)

    # Load the variables of each model with each other's values.
    load_tf_model(path1, tfic2_vars)
    assert tf_var_equal(tfic1_vars, tfic2_vars)
    load_tf_model(path2, tfic1_vars)
    assert not tf_var_equal(tfic1_vars, tfic2_vars)

    with raises(TypeError):
        save_tf_model(1, tfic1_vars)

    with raises(TypeError):
        save_tf_model(path1, 1)

    with raises(ValueError):
        save_tf_model(path1, [])

    with raises(TypeError):
        load_tf_model(1, tfic1_vars)

    with raises(TypeError):
        load_tf_model(path1, 1)

    with raises(ValueError):
        load_tf_model(path1, [])

    with raises(ValueError):
        load_tf_model("foo", tfic1_vars)

    x = tf.Variable(initial_value=np.random.normal(0, 1, (2, 2)), name="x")
    with raises(ValueError):
        load_tf_model(path1, [x])

    return None
