import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
from svgd import SVGD


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


def inference(inputs, labels, model):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        log_likelihood = -1 * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)

    prob_1_x_w = tf.nn.sigmoid(logits)
    gradients = tape.gradient(log_likelihood, model.trainable_variables)

    return gradients, model.trainable_variables, prob_1_x_w


def predict(inputs, model):
    logits = model(inputs)
    prob_1_x_w = tf.nn.sigmoid(logits)

    return logits, prob_1_x_w


if __name__ == "__main__":

    N = 400
    num_particles = 100
    num_iterations = 250

    X_train, X_test, y_train, y_test = get_toy_data_2d(
        n_samples=N, test_size=0.25)

    grad_optimizer = tf.optimizers.Adagrad(learning_rate=0.05)

    # initialization
    models = []
    for i in range(num_particles):
        models.append(tf.keras.layers.Dense(1,
                                            kernel_initializer=tf.keras.initializers.RandomNormal(
                                                mean=0.0, stddev=0.8),
                                            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.8)))

    # train
    for _ in range(num_iterations):
        grads_list, vars_list, prob_1_x_w_list = [], [], []

        for i in range(num_particles):
            grads, variables, prob_1_x_w = inference(
                X_train, y_train, models[i])
            grads_list.append(grads)
            vars_list.append(variables)
            prob_1_x_w_list.append(prob_1_x_w)

        svgd = SVGD(grads_list=grads_list,
                    vars_list=vars_list,
                    optimizer=grad_optimizer)
        svgd.run()

    # evaluation
    # train
    prob_train_x = tf.reduce_mean(tf.stack(prob_1_x_w_list), axis=0)
    classification = prob_train_x.numpy() > 0.5
    accuracy = np.sum(classification == y_train) / y_train.shape[0]
    print("train accuracy score: {:.2f}".format(accuracy))

    # test
    prob_test_list = []
    for i in range(num_particles):
        _, prob_test = predict(X_test, models[i])
        prob_test_list.append(prob_test)

    prob_test_x = tf.reduce_mean(tf.stack(prob_test_list), axis=0)
    classification = prob_test_x.numpy() > 0.5
    accuracy = np.sum(classification == y_test) / y_test.shape[0]
    print("test accuracy score: {:.2f}".format(accuracy))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    x0_grid, x1_grid = np.linspace(-7, 7, 50), np.linspace(-7, 7, 50)
    x0_grid, x1_grid = np.meshgrid(x0_grid, x1_grid)

    x_grid = np.hstack([x0_grid.reshape(-1, 1), x1_grid.reshape(-1, 1)])
    prob_grids = []
    for i in range(num_particles):
        _, prob_grid = predict(x_grid, models[i])
        prob_grids.append(prob_grid)

    probs = tf.reduce_mean(tf.stack(prob_grids), axis=0)
    probs = probs.numpy().reshape(x0_grid.shape)

    contour = ax.contour(x0_grid, x1_grid, probs, 50,
                         cmap=plt.cm.coolwarm, zorder=0)
    x0, x1 = X_train[np.where(y_train[:, 0] == 0)
                     ], X_train[np.where(y_train[:, 0] == 1)]
    ax.scatter(x0[:, 0], x0[:, 1], s=5, c='blue', zorder=1)
    ax.scatter(x1[:, 0], x1[:, 1], s=5, c='red', zorder=2)

    ax.set_title("Bayesian Logistic Regression with SVGD")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', 'box')
    fig.colorbar(contour)
    plt.savefig("blr_result.png")
