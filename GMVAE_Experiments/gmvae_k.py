import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sys
from gmvae_model import GMVAE
from utils import *

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
def main():
    try:
        k = int(sys.argv[1])
    except IndexError:
        k = 2
        print('Setting default value k={0}'.format(k))

    n_x = 10
    n_z = 2

    dataset = load_and_mix_data('generated_from_cluster',k,True)

    model = GMVAE(k=k, n_x=n_x, n_z=n_z)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=4)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters from file (optional)
        #saver.restore(sess, './savedModels/2018-2-22/model-30')

        # TRAINING
        sess_info = (sess, saver)
        model.train('logs/gmvae_k={:d}.log'.format(k), dataset, sess_info, epochs=10)

        # SCATTER PLOT
        plot_z(sess,
               dataset.test.data,
               dataset.test.labels,
               model,
               k,
               n_z)
        plot_z_means(sess,
                     dataset.test.data,
                     dataset.test.labels,
                     model,
                     k,
                     n_z)
        plot_gmvae_output(sess,
                          dataset.test.data,
                          dataset.test.labels,
                          model,
                          k)

        sample_and_plot_z(sess, k, model, 300)
        sample_and_plot_x(sess, k, model, 300)



main()
