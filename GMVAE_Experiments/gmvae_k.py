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

    n_x = 2
    n_z = 2

    dataset = load_and_mix_data('generated_from_cluster',k,True)
    if True:
        x = dataset.test.data
        y = dataset.test.labels
        print('Plotting dataset variables')
        plot_labeled_data(x, np.argmax(y, axis=1), 'scatter_x.png')

    model = GMVAE(k=k, n_x=n_x, n_z=n_z)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=4, max_to_keep=20)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        use_pretrained = False
        if use_pretrained:
            # Restore parameters from file (optional)
            saver.restore(sess, './savedModels/2018-3-22/model-100')
        else:
            # TRAINING
            sess_info = (sess, saver)
            history = model.train('logs/gmvae_k={:d}.log'.format(k), dataset, sess_info, epochs=100, is_labeled = True)

            plot_lines(history['iters'], [history['loss'], history['val_loss']], 'Loss')
            plot_lines(history['iters'], [history['val_acc']], 'Accuracy')

        # SCATTER PLOT
        y_pred = plot_z(sess,
               dataset.test.data,
               np.argmax(dataset.test.labels, axis=1),
               model,
               k,
               n_z)
        #print(y_pred)
        plot_z_means(sess,
               dataset.test.data,
               np.argmax(dataset.test.labels, axis=1),
               model,
               k,
               n_z)

        if use_pretrained:
            plot_gmvae_output(sess,
                              dataset.test.data,
                              np.argmax(dataset.test.labels, axis=1),
                              model,
                              k)

            sample_and_plot_z(sess, k, model, 300)
            sample_and_plot_x(sess, k, model, 300)



main()
