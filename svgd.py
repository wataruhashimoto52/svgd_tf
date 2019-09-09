import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp


class SVGD(object):
    def __init__(self, grads_list, vars_list, optimizer):
        self.grads_list = grads_list
        self.vars_list = vars_list
        self.optimizer = optimizer
        self.num_particles = len(vars_list)

    def get_pairwise_dist(self, x):
        norm = tf.reshape(tf.reduce_sum(x * x, 1), [-1, 1])
        return norm - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(norm)

    def _get_svgd_kernel(self, X):
        stacked_vars = tf.stack(X)
        pairwise_dists = self.get_pairwise_dist(stacked_vars)
        lower = tfp.stats.percentile(
            pairwise_dists, 50.0, interpolation='lower')
        higher = tfp.stats.percentile(
            pairwise_dists, 50.0, interpolation='higher')

        median = (lower + higher) / 2
        median = tf.cast(median, tf.float32)
        h = tf.sqrt(0.5 * median / tf.math.log(len(X) + 1.))
        h = tf.stop_gradient(h)

        # kernel computation
        Kxy = tf.exp(-pairwise_dists / h ** 2 / 2)
        dxkxy = -tf.matmul(Kxy, stacked_vars)
        sumkxy = tf.reduce_sum(Kxy, axis=1, keepdims=True)

        # analytical kernel gradient
        dxkxy = (dxkxy + stacked_vars * sumkxy) / tf.pow(h, 2)

        return Kxy, dxkxy

    def get_num_elements(self, var):
        return int(np.prod(self.var_shape(var)))

    def flatten_grads_and_vars(self, grads, variables):
        # from openai/baselines/common/tf_util.py
        flatgrads = tf.concat(axis=0, values=[
            tf.reshape(grad if grad is not None else tf.zeros_like(var), [self.get_num_elements(var)]) for (var, grad) in zip(variables, grads)])
        flatvars = tf.concat(axis=0, values=[
            tf.reshape(var, [self.get_num_elements(var)])for var in variables])
        return flatgrads, flatvars

    def var_shape(self, var):
        out = var.get_shape().as_list()
        return out

    def run(self):
        flatgrads_list, flatvars_list = [], []

        for grads, variables in zip(self.grads_list, self.vars_list):
            flatgrads, flatvars = self.flatten_grads_and_vars(grads, variables)
            flatgrads_list.append(flatgrads)
            flatvars_list.append(flatvars)

        Kxy, dxkxy = self._get_svgd_kernel(flatvars_list)
        stacked_grads = tf.stack(flatgrads_list)
        stacked_grads = (tf.matmul(Kxy, stacked_grads) -
                         dxkxy) / self.num_particles
        flatgrads_list = tf.unstack(stacked_grads, self.num_particles)

        # align index
        grads_list = []
        for flatgrads, variables in zip(flatgrads_list, self.vars_list):
            start = 0
            grads = []

            for var in variables:
                shape = self.var_shape(var)
                end = start + int(np.prod(self.var_shape(var)))
                grads.append(tf.reshape(flatgrads[start:end], shape))
                start = end

            grads_list.append(grads)

        for grads, variables in zip(grads_list, self.vars_list):
            self.optimizer.apply_gradients(
                [(-grad, var) for grad, var in zip(grads, variables)])

        return
