import tensorflow as tf
import numpy as np
from tensorbayes.layers import constant, placeholder
from subgraphs import z_graph, px_graph, px_fixed_graph
from tensorbayes.nbutils import show_default_graph
from utils import get_var, plot_labeled_data

k = 3
n_x = 10
sample_size = 500
tf.reset_default_graph()

#with tf.name_scope('y_'):
 #   y_ = tf.fill(tf.stack([None, k]), 0.0)

# for each proposed y, infer z and reconstruct x
zm_prior, \
zv_prior, \
z, \
x = [[None] * k for i in range(4)]
zm_prior[0] = constant(np.ones((sample_size,2)))
zv_prior[0] = constant(np.ones((sample_size,2)))
zm_prior[1] = constant(-1*np.ones((sample_size,2)))
zv_prior[1] = constant(np.ones((sample_size,2)))
zm_prior[2] = constant(np.stack([np.array([3,-1]) for i in range(sample_size)]))
zv_prior[2] = constant(np.ones((sample_size,2)))*0.5
for i in range(k):
    with tf.name_scope('graphs/hot_at{:d}'.format(i)):
        z[i] = z_graph(zm_prior[i], zv_prior[i])
        x[i] = px_fixed_graph(z[i], n_x)

show_default_graph()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_0,x_1,x_2,z0,z1,z2  = sess.run([x[0], x[1], x[2], z[0], z[1], z[2]])
    np.save('./generatedData/generated_from_cluster0.npy', x_0)
    np.save('./generatedData/generated_from_cluster1.npy', x_1)
    np.save('./generatedData/generated_from_cluster2.npy', x_2)

    y_0 = np.full((len(z0)), 0)
    y_1 = np.full((len(z1)), 1)
    y_2 = np.full((len(z2)), 2)

    z = np.concatenate((z0, z1, z2))
    x = np.concatenate(((x_0, x_1, x_2)))
    y = np.concatenate((y_0, y_1, y_2))
    print('Plotting generated latent variables')
    plot_labeled_data(z, y, 'scatter_true_z.png')
    #print('Plotting generated variables')
    #plot_labeled_data(x, y, 'scatter_x.png')

    # weights
    #vars = get_var('px/layer1/weights')
    #print('layer1: \n {}'.format(sess.run(vars)))
    #vars = get_var('px/output/weights')
    #print('output: \n {}'.format(sess.run(vars)))