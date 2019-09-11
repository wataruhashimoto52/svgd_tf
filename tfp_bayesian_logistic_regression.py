import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
from toy_data import get_toy_data_2d

model = tfp.layers.DenseFlipout(units=1,
                                activation=None,
                                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn())


optimizer = tf.optimizers.Adagrad(learning_rate=0.05)


def inference(inputs, labels, N):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        labels_distribution = tfp.distributions.Bernoulli(logits=logits)
        neg_log_likelihood = - \
            tf.reduce_mean(input_tensor=labels_distribution.log_prob(labels))
        kl = sum(model.losses) / N
        elbo_loss = neg_log_likelihood + kl
    gradients = tape.gradient(elbo_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test_inference(inputs):
    logits = model(inputs)
    labels_distribution = tfp.distributions.Bernoulli(logits=logits)

    return logits, labels_distribution


if __name__ == "__main__":

    N = 400
    num_particle = 100
    num_iterations = 250
    X_train, X_test, y_train, y_test = get_toy_data_2d(
        n_samples=N, test_size=0.25)

    # train
    for epoch in range(num_iterations):
        inference(X_train, y_train, N)

    # evaluation
    # train
    logits, labels_dist = test_inference(X_train)
    ensembled = tf.reduce_sum(labels_dist.sample(
        num_particle), axis=0, keepdims=False) / num_particle
    classification = ensembled.numpy() > 0.5
    accuracy = np.sum(classification == y_train) / y_train.shape[0]
    print("train accuracy score: {:.2f}".format(accuracy))

    # test
    logits_pred, labels_dist_pred = test_inference(X_test)
    ensembled_pred = tf.reduce_sum(labels_dist_pred.sample(
        num_particle), axis=0, keepdims=False) / num_particle
    classification = ensembled_pred.numpy() > 0.5
    accuracy = np.sum(classification == y_test) / y_test.shape[0]
    print("test accuracy score: {:.2f}".format(accuracy))

    # plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    x0_grid, x1_grid = np.linspace(-7, 7, 50), np.linspace(-7, 7, 50)
    x0_grid, x1_grid = np.meshgrid(x0_grid, x1_grid)
    x_grid = np.hstack([x0_grid.reshape(-1, 1), x1_grid.reshape(-1, 1)])

    _, labels_dist_grid = test_inference(x_grid)
    ensembled_grid = tf.reduce_sum(labels_dist_grid.sample(
        num_particle), axis=0, keepdims=False) / num_particle
    probs = ensembled_grid.numpy().reshape(x0_grid.shape)

    contour = ax.contour(x0_grid, x1_grid, probs, 50,
                         cmap=plt.cm.coolwarm, zorder=0)
    x0, x1 = X_train[np.where(y_train[:, 0] == 0)
                     ], X_train[np.where(y_train[:, 0] == 1)]
    ax.scatter(x0[:, 0], x0[:, 1], s=5, c='blue', zorder=1)
    ax.scatter(x1[:, 0], x1[:, 1], s=5, c='red', zorder=2)

    ax.set_title("Bayesian Logistic Regression with MFVI")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', 'box')
    fig.colorbar(contour)
    plt.savefig("tfp_blr_result.png")
