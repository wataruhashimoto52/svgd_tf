import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp


class SVGD(object):
    def __init__(self, grads_list, vars_list, optimizer):
        self.grads_list = grads_list
        self.vars_list = vars_list
        self.optimizer = optimizer
        self.num_particles = len(vars_list)
        self.update_op = self.build_optimizer()

    def _get_svgd_kernel(self, flatvars_list):
        stacked_vars = tf.stack(flatvars_list)
        norm = tf.reduce_sum(stacked_vars*stacked_vars, 1)
        norm = tf.reshape(norm, [-1, 1])
        pairwise_dists = norm - 2 * \
            tf.matmul(stacked_vars, tf.transpose(
                stacked_vars)) + tf.transpose(norm)

        lower = tfp.stats.percentile(
            pairwise_dists, 50.0, interpolation='lower')
        higher = tfp.stats.percentile(
            pairwise_dists, 50.0, interpolation='higher')

        median = (lower + higher) / 2
        median = tf.cast(median, tf.float32)
        h = tf.sqrt(0.5 * median / tf.math.log(len(flatvars_list) + 1.))
        h = tf.stop_gradient(h)

        if len(flatvars_list) == 1:
            h = 1.

        # kernel computation
        Kxy = tf.exp(- pairwise_dists / h ** 2 / 2)
        dxkxy = - tf.matmul(Kxy, stacked_vars)
        sumkxy = tf.reduce_sum(Kxy, axis=1, keepdims=True)

        # analytical kernel gradient
        dxkxy = (dxkxy + stacked_vars * sumkxy) / tf.pow(h, 2)

        return (Kxy, dxkxy)

    def get_num_elements(self, var):
        return int(np.prod(self.var_shape(var)))

    def flatten_grads_and_vars(self, grads, vars):
        # from openai/baselines/common/tf_util.py
        flatgrads = tf.concat(axis=0, values=[
            tf.reshape(grad if grad is not None else tf.zeros_like(
                var), [self.get_num_elements(var)])
            for (var, grad) in zip(vars, grads)])
        flatvars = tf.concat(axis=0, values=[
            tf.reshape(var, [self.get_num_elements(var)])
            for var in vars])
        return flatgrads, flatvars

    @staticmethod
    def var_shape(var):
        out = var.get_shape().as_list()
        assert all(isinstance(a, int) for a in out), \
            'shape function assumes that shape is fully known'
        return out

    def build_optimizer(self):
        flatgrads_list, flatvars_list = [], []

        for grads, vars in zip(self.grads_list, self.vars_list):
            flatgrads, flatvars = self.flatten_grads_and_vars(grads, vars)
            flatgrads_list.append(flatgrads)
            flatvars_list.append(flatvars)

        Kxy, dxkxy = self._get_svgd_kernel(flatvars_list)
        stacked_grads = tf.stack(flatgrads_list)
        stacked_grads = (tf.matmul(Kxy, stacked_grads) +
                         dxkxy) / self.num_particles
        flatgrads_list = tf.unstack(stacked_grads, self.num_particles)

        grads_list = []
        for flatgrads, variables in zip(flatgrads_list, self.vars_list):
            start = 0
            grads = []

            for var in variables:
                shape = self.var_shape(var)
                size = int(np.prod(shape))

                end = start + size
                grads.append(tf.reshape(flatgrads[start:end], shape))
                start = end

            grads_list.append(grads)

        update_ops = []
        for grads, variables in zip(grads_list, self.vars_list):
            opt = self.optimizer
            update_ops.append(opt.apply_gradients(
                [(-g, v) for g, v in zip(grads, variables)]))

        return tf.group(*update_ops)
