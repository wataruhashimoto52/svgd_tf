import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from svgd import SVGD


def generate_data(n_samples, test_size=0.25):
    num0 = n_samples // 2
    num1 = 400 - num0

    mean0 = np.array([-1, -1])
    std0 = np.array([0.25, 0.25])
    mean1 = np.array([1, 1])
    std1 = np.array([1.5, 1.5])

    x0 = np.tile(mean0, (num0, 1)) + std0 * np.random.randn(num0, 2)
    x1 = np.tile(mean1, (num1, 1)) + std1 * np.random.randn(num1, 2)
    y0 = np.zeros((x0.shape[0], 1))
    y1 = np.ones((x1.shape[0], 1))

    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([y0, y1], axis=0)
    D = np.hstack([x, y])
    np.random.shuffle(D)
    x = np.array(D[:, 0:2], dtype=np.float32)
    y = np.array(D[:, 2:], dtype=np.float32)
    train_size = int(n_samples * test_size)
    X_train = x[:train_size]
    y_train = y[:train_size]
    X_test = x[train_size:]
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
    D = 2
    num_particles = 20
    num_iterations = 100


    X_train, X_test, y_train, y_test = generate_data(n_samples=N, test_size=0.25)

    grad_optimizer = tf.optimizers.Adagrad(learning_rate=0.05)

    # initialization
    models = []
    for i in range(num_particles):
        models.append(tf.keras.layers.Dense(1))

    # train
    for _ in range(num_iterations):
        grads_list, vars_list, prob_1_x_w_list = [], [], []
        
        for i in range(num_particles):
            grads, variables, prob_1_x_w = inference(X_train, y_train, models[i])
            grads_list.append(grads)
            vars_list.append(variables)
            prob_1_x_w_list.append(prob_1_x_w)

        optimizer = SVGD(grads_list=grads_list,
                        vars_list=vars_list,
                        optimizer=grad_optimizer)
        optimizer.update_op


    # evaluation
    ## train
    prob_train_x = tf.reduce_mean(tf.stack(prob_1_x_w_list), axis=0)
    classification = prob_train_x.numpy() > 0.5
    accuracy = np.sum(classification == y_train) / y_train.shape[0]
    print("train accuracy score: {:.2f}".format(accuracy))


    ## test
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

    contour = ax.contour(x0_grid, x1_grid, probs, 50, cmap=plt.cm.coolwarm, zorder=0)
    x0, x1 = X_train[np.where(y_train[:, 0] == 0)], X_train[np.where(y_train[:, 0] == 1)]
    ax.scatter(x0[:, 0], x0[:, 1], s=1, c='blue', zorder=1)
    ax.scatter(x1[:, 0], x1[:, 1], s=1, c='red', zorder=2)

    ax.set_title('$p(1|(x_0, x_1))$ with {} ({} particles)'.format("svgd", num_particles))
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', 'box')
    ax.grid(b=True)
    fig.colorbar(contour)
    plt.savefig("blr_result.png")
