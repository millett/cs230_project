import tensorflow as tf
import numpy as np
from tensorbayes.layers import constant, placeholder
from subgraphs import qy_graph, qz_graph, px_graph, pz_graph, labeled_loss
from utils import open_file, progbar, stream_print, test_acc, save_params, get_var
from tensorbayes.nbutils import show_default_graph

class GMVAE():
    def __init__(self, k=10, n_x=784, n_z = 64):
        self.k = k
        self.n_x = n_x
        self.n_z = n_z
        tf.reset_default_graph()
        x = placeholder((None, n_x), name='x')

        # create a y "placeholder"
        with tf.name_scope('y_'):
            y_ = tf.fill(tf.stack([tf.shape(x)[0], k]), 0.0)

        # propose distribution over y
        self.qy_logit, self.qy = qy_graph(x, k)

        # for each proposed y, infer z and reconstruct x
        self.z, \
        self.zm, \
        self.zv, \
        self.zm_prior, \
        self.zv_prior, \
        self.px_logit,\
        self.y = [[None] * k for i in range(7)]
        for i in range(k):
            with tf.name_scope('graphs/hot_at{:d}'.format(i)):
                y = tf.add(y_,
                           constant(np.eye(k)[i], name='hot_at_{:d}'.format(i)))
                self.z[i], self.zm[i], self.zv[i] = qz_graph(x, y, n_z)
                self.y[i], \
                self.zm_prior[i], \
                self.zv_prior[i] = pz_graph(y, n_z)
                self.px_logit[i] = px_graph(self.z[i], n_x)

        # Aggressive name scoping for pretty graph visualization :P
        with tf.name_scope('loss'):
            with tf.name_scope('neg_entropy'):
                self.nent = -tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.qy,
                    logits=self.qy_logit)
            losses = [None] * k
            for i in range(k):
                with tf.name_scope('loss_at{:d}'.format(i)):
                    losses[i] = labeled_loss(x, self.px_logit[i], self.z[i], self.zm[i],
                                             self.zv[i], self.zm_prior[i], self.zv_prior[i])
            with tf.name_scope('final_loss'):
                self.loss = tf.add_n(
                    [self.nent] + [self.qy[:, i] * losses[i] for i in range(k)])

        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        show_default_graph()

    def train(self, fname, dataset, sess_info, epochs):
        (sess, saver) = sess_info
        f = open_file(fname)
        iterep = 500
        for i in range(iterep * epochs):
            batch = dataset.train.next_batch(100)
            sess.run(self.train_step,
                     feed_dict={'x:0': batch})

            progbar(i, iterep)
            if (i + 1) % iterep == 0:
                a, b = sess.run([self.nent, self.loss], feed_dict={
                    'x:0': dataset.train.data[np.random.choice(len(dataset.train.data),
                                                               200)]})
                c, d = sess.run([self.nent, self.loss],
                                feed_dict={'x:0': dataset.test.data})
                a, b, c, d = -a.mean(), b.mean(), -c.mean(), d.mean()
                e = test_acc(dataset, sess, self.qy_logit)
                string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
                          .format('tr_ent', 'tr_loss', 't_ent', 't_loss',
                                  't_acc', 'epoch'))
                stream_print(f, string, i <= iterep)
                string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
                          .format(a, b, c, d, e, int((i + 1) / iterep)))
                stream_print(f, string)
            # Saves parameters every 10 epochs
            if (i + 1) % (10 * iterep) == 0:
                print('saving')
                save_params(saver, sess, (i + 1) // iterep)
        if f is not None: f.close()