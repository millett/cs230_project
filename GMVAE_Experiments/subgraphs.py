import tensorflow as tf
from tensorbayes.layers import gaussian_sample
from tensorbayes.distributions import log_bernoulli_with_logits, log_normal, log_squared_loss
import numpy as np

# n_x is number input features
# n_z is dim of latent space
# k is number of clusters/categories in gaussian mix

n_h=16
use_batch_norm = False

# vae subgraphs
def qy_graph(x, k, phase):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qy')) > 0
    # -- q(y)
    with tf.variable_scope('qy'):
        h1 = tf.contrib.layers.fully_connected(x, n_h, scope='layer1',
                                               activation_fn=tf.nn.relu,
                                               reuse=reuse)
        if use_batch_norm:
            h1 = tf.contrib.layers.batch_norm(h1,
                                              center=True, scale=True,
                                              is_training=phase, reuse=reuse,
                                              scope='bn1')
        h2 = tf.contrib.layers.fully_connected(h1, n_h, scope='layer2',
                                               activation_fn=tf.nn.relu,
                                               reuse=reuse)
        if use_batch_norm:
            h2 = tf.contrib.layers.batch_norm(h2,
                                              center=True, scale=True,
                                              is_training=phase, reuse=reuse,
                                              scope='bn2')
        qy_logit = tf.contrib.layers.fully_connected(h2, k, scope='logit',
                                               activation_fn=tf.nn.relu, reuse=reuse)
        qy = tf.nn.softmax(qy_logit, name='prob')
    return qy_logit, qy

def qz_graph(x, y, n_z, phase):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qz')) > 0
    # -- q(z)
    with tf.variable_scope('qz'):
        h0 = tf.contrib.layers.fully_connected(y, int(y.get_shape()[-1]), scope='layer0',
                                               activation_fn=None,
                                               reuse=reuse)
        xy = tf.concat((x, h0), 1, name='xy/concat')
        h1 = tf.contrib.layers.fully_connected(xy, n_h, scope='layer1',
                                               activation_fn=tf.nn.relu,
                                               reuse=reuse)
        if use_batch_norm:
            h1 = tf.contrib.layers.batch_norm(h1,
                                         center=True, scale=True,
                                         is_training=phase, reuse=reuse,
                                         scope='bn1')
        h2 = tf.contrib.layers.fully_connected(h1, n_h, scope='layer2',
                                               activation_fn=tf.nn.relu,
                                               reuse=reuse)
        if use_batch_norm:
            h2 = tf.contrib.layers.batch_norm(h2,
                                              center=True, scale=True,
                                              is_training=phase, reuse=reuse,
                                              scope='bn2')
        zm = tf.contrib.layers.fully_connected(h2, n_z, scope='zm',
                                               activation_fn=None,
                                               reuse=reuse)
        zv = tf.contrib.layers.fully_connected(h2, n_z, scope='zv',
                                               activation_fn=tf.nn.softplus,
                                               reuse=reuse)
        z = z_graph(zm,zv)
    return z, zm, zv

def z_graph(zm,zv):
    with tf.variable_scope('z'):
        z = gaussian_sample(zm, zv, 'z')
        # Used to feed into z when sampling
        z = tf.identity(z, name='z_sample')
    return z

def pz_graph(y, n_z, phase):
    reuse = len(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pz')) > 0
    # -- p(z)
    with tf.variable_scope('pz'):
        h1 = tf.contrib.layers.fully_connected(y, n_h, scope='layer1',
                                               # 4 = n_x + k. MIMIC is more complex, will add more layers
                                               activation_fn=tf.nn.relu,
                                               reuse=reuse)
        if use_batch_norm:
            h1 = tf.contrib.layers.batch_norm(h1,
                                              center=True, scale=True,
                                              is_training=phase, reuse=reuse,
                                              scope='bn1')
        zm = tf.contrib.layers.fully_connected(h1, n_z, scope='zm',
                                               activation_fn=None, reuse=reuse)
        zv = tf.contrib.layers.fully_connected(h1, n_z, scope='zv',
                                               activation_fn=tf.nn.softplus,
                                               reuse=reuse)
    return y, zm, zv

def px_fixed_graph(z, n_x):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='px_fixed')) > 0
    # -- p(x)
    with tf.variable_scope('px_fixed'):
        h = tf.contrib.layers.fully_connected(z, n_h, scope='layer1',
                                    activation_fn=tf.nn.relu,
                                    reuse=reuse)
        px_logit = tf.contrib.layers.fully_connected(h, n_x, scope='output',
                                                     activation_fn=None,
                                                     reuse=reuse)
        #px_logit = tf.identity(px_logit,name='x')
    return px_logit

def px_graph(z, n_x, phase):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='px')) > 0
    # -- p(x)
    with tf.variable_scope('px'):
        h1 = tf.contrib.layers.fully_connected(z, n_h, scope='layer1',
                                               activation_fn=tf.nn.relu,
                                               reuse=reuse)
        if use_batch_norm:
            h1 = tf.contrib.layers.batch_norm(h1,
                                              center=True, scale=True,
                                              is_training=phase, reuse=reuse,
                                              scope='bn1')
        h2 = tf.contrib.layers.fully_connected(h1, n_h, scope='layer2',
                                               activation_fn=tf.nn.relu,
                                               reuse=reuse)
        if use_batch_norm:
            h2 = tf.contrib.layers.batch_norm(h2,
                                              center=True, scale=True,
                                              is_training=phase, reuse=reuse,
                                              scope='bn2')
        xm = tf.contrib.layers.fully_connected(h2, n_x, scope='xm',
                                                     activation_fn=None,
                                                     reuse=reuse)
        xv = tf.contrib.layers.fully_connected(h2, n_x, scope='xv',
                                                    activation_fn=tf.nn.softplus,
                                                     reuse=reuse)
        #px_logit = tf.identity(px_logit,name='x')
    return xm, xv


def labeled_loss(x, xm, xv, z, zm, zv, zm_prior, zv_prior):
    xy_loss = -log_normal(x, xm, xv)
    xy_loss += log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
    return xy_loss - np.log(0.0001)
