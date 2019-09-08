import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def rbf_kernel(dist, h):
    return tf.exp(-1 * dist / h)


def get_pairwise_dist(x):
    norm = tf.reshape(tf.reduce_sum(x * x, 1), [-1, 1])
    return norm - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(norm)


def get_median(x, h):
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0] // 2
    median = tf.nn.top_k(v, m).values[m-1]
    median = tf.cast(median, tf.float32)
    
    h = tf.sqrt(0.5 * median / tf.math.log(x.shape[0] + 1.))
    h = tf.stop_gradient(h)
    
    
    return h


def get_svgd_kernel(X0):
    XY = tf.matmul(X0, tf.transpose(X0))
    X2_ = tf.reduce_sum(tf.square(X0), axis=1)

    x2 = tf.reshape( X2_, shape=( tf.shape(X0)[0], 1) )
    
    X2e = tf.tile(x2, [1, tf.shape(X0)[0] ] )
    H = tf.subtract(tf.add(X2e, tf.transpose(X2e) ), 2 * XY)

    V = tf.reshape(H, [-1,1]) 
    
    # median distance
    def get_median(v):
        v = tf.reshape(v, [-1])
        m = v.get_shape()[0]//2
        return tf.nn.top_k(v, m).values[m-1]
    h = get_median(V)
    h = tf.sqrt(0.5 * h / tf.math.log( tf.cast( tf.shape(X0)[0] , tf.float32) + 1.0))
    

    return tf.exp(-H / h ** 2 / 2.0)


    
    
@tf.function
def get_phi(samples, lnprob, num_particles, dim, h=1.0):
    samples = tf.reshape(samples, (num_particles, dim))
    kernel_matrix = get_svgd_kernel(samples)
    kernel_grad = tf.gradients(kernel_matrix, samples)
    
    # get log-probability gradients
    log_prob = lnprob(samples)
    log_prob_grad = tf.gradients(log_prob, samples)
    
    operation = tf.matmul(kernel_matrix, log_prob_grad) - kernel_grad
    operation = tf.reshape(operation, (num_particles, dim))
    
    
    return operation / num_particles


if __name__ == "__main__":
    eps = 0.1
    num_particles = 400
    num_iter = 200
    dim = 2

    tfd = tfp.distributions

    gmm = tfd.Mixture(
            cat=tfd.Categorical(probs=[0.5, 0.5]),
            components=[tfd.MultivariateNormalDiag(loc=tf.constant([-3., +3]), scale_diag=tf.constant([1., 1.])),
                        tfd.MultivariateNormalDiag(loc=tf.constant([+3., -3]), scale_diag=tf.constant([1., 1.]))
                        ])


    q_init = tfd.MultivariateNormalDiag(loc=tf.constant([0., 0.]),
                                            scale_diag=tf.constant([2., 2.]))

    samples = q_init.sample(num_particles)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()

    artists = []

    for i in range(num_iter):
        
        grads = get_phi(samples, gmm.log_prob, num_particles, dim)
        samples = samples + eps * grads

        artist = ax.scatter(samples[:, 0].numpy(),
                            samples[:, 1].numpy(),
                            marker="o",
                            c="b")

        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-7, 7)
        ax.set_xlim(-7, 7)
        
        artists.append([artist])
        
        
    ani = animation.ArtistAnimation(fig,
                                    artists,
                                    interval=100,
                                    blit=True
                                    )

    ani.save("svgd_gmm.gif", writer="imagemagick", fps=15)
