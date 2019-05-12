import argparse
import json
import os
import numpy as np
import tensorflow as tf
from gpt_2_simple.src import memory_saving_gradients
import time


class AccumulatingOptimizer(object):
    def __init__(self, opt, var_list, use_memory_saving_gradients=False):
        self.opt = opt
        self.var_list = var_list
        self.accum_vars = {tv : tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
                           for tv in var_list}
        self.total_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))
        self.count_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))
        self.use_memory_saving_gradients = use_memory_saving_gradients

    def reset(self):
        updates = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars.values()]
        updates.append(self.total_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        updates.append(self.count_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def compute_gradients(self, loss):
        if self.use_memory_saving_gradients:
            grads = memory_saving_gradients.gradients(loss, self.var_list)
        else:
            grads = self.opt.compute_gradients(loss, self.var_list)
        updates = [self.accum_vars[v].assign_add(g) for (g,v) in grads]
        updates.append(self.total_loss.assign_add(loss))
        updates.append(self.count_loss.assign_add(1.0))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def apply_gradients(self):
        grads = [(g,v) for (v,g) in self.accum_vars.items()]
        with tf.control_dependencies([self.opt.apply_gradients(grads)]):
            return self.total_loss / self.count_loss
