import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def get_toy_data_2d(n_samples=400, test_size=0.25):
    n_class0 = n_samples // 2
    n_class1 = n_samples - n_class0

    class0_dist = tfp.distributions.MultivariateNormalDiag(loc=tf.constant([-1., -1.]),
                                                               scale_diag=tf.constant([0.25, 0.25]))
    class1_dist = tfp.distributions.MultivariateNormalDiag(loc=tf.constant([1., 1.]),
                                                               scale_diag=tf.constant([1.5, 1.5]))

    class0_samples = class0_dist.sample(n_class0)
    y0 = tf.zeros((n_class0, 1))
    class1_samples = class1_dist.sample(n_class1)
    y1 = tf.ones((n_class1, 1))

    X = tf.concat([class0_samples, class1_samples], axis=0)
    y = tf.concat([y0, y1], axis=0)
    data = tf.concat([X, y], axis=1)
    data = tf.random.shuffle(data, seed=123)
    X = data[:, 0:2].numpy().astype(np.float32)
    y = data[:, 2:].numpy().astype(np.float32)
    train_size = int(n_samples * (1 - test_size))
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test
