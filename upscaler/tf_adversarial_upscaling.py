import logging
import os

import luigi
import matplotlib.pyplot as plt
import numpy as np
from neobunch import Bunch
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib import slim

import upscaler.data


class TFAdversarialUpscaling(luigi.Task):
    box_size = luigi.IntParameter(default=16)
    remove_size = luigi.IntParameter(default=4)
    batch_size = 128

    def requires(self):
        return upscaler.data.ImageReader(box_size=self.box_size, remove_size=self.remove_size)

    def batch_iterator(self, data):
        di = self.data_info(data)
        permutation = np.random.permutation(di.n)

        for i in range(0, di.n-self.batch_size, self.batch_size):
            batch_ix = permutation[i:i+self.batch_size]
            batch = data[batch_ix].copy()
            X = batch.copy()
            X[:, :, di.l:di.r, di.l:di.r] = 0
            y = batch[:, :, di.l:di.r, di.l:di.r]

            yield X.transpose(0, 2, 3, 1), y.transpose(0, 2, 3, 1)

    def img_dump(self, enc, y, X, l, r):
        logging.warning('Writing images')
        for i in tqdm(range(16)):
            xim = X[i]
            plt.imshow(xim)
            plt.savefig('images/%02d-a.png' % i)
            plt.close()
            im = enc[i]
            expected = y[i]
            xim[l:r, l:r, :] = im
            plt.imshow(xim)
            plt.savefig('images/%02d-b.png' % i)
            plt.close()
            xim[l:r, l:r, :] = expected
            plt.imshow(xim)
            plt.savefig('images/%02d-c.png' % i)
            plt.close()

    def data_info(self, data):
        n, c, w, h = data.shape
        center = w // 2
        l, r = center-self.remove_size, center+self.remove_size
        return Bunch(
            n=n, c=c, w=w, h=h, l=l, r=r, center=center
        )

    def run(self):
        data = self.requires().load()
        di = self.data_info(data)
        graph = tf.Graph()
        with graph.as_default():
            input_t = tf.placeholder(tf.float32, [None, 16, 16, 3])
            y_t = tf.placeholder(tf.float32, (None, 8, 8, 3))
            conv1 = slim.conv2d(input_t, 64, 3, padding='VALID')
            conv2 = slim.conv2d(conv1, 128, 3, padding='VALID')
            conv3 = slim.conv2d(conv2, 128, 5, padding='VALID')
            conv4 = slim.conv2d(conv3, 3, 3, padding='SAME')


            top =    input_t[:, :di.l,      : :]
            left =   input_t[:, di.l:di.r, :di.l, :]
            right =  input_t[:, di.l:di.r, di.r:, :]
            bottom = input_t[:, di.r:,     :, :]

            middle = tf.concat([left, conv4, right], 2)
            full = tf.concat([top, middle, bottom], 1)
            loss = tf.losses.mean_squared_error(y_t, conv4)
            opt = tf.train.AdamOptimizer(0.001).minimize(loss)

            with tf.Session() as ses:
                ses.run(tf.global_variables_initializer())
                for x, y in self.batch_iterator(data):
                    print(ses.run([loss, opt], {input_t: x, y_t: y}))

                    if np.random.uniform() < 0.1:
                        enc = ses.run(conv4, {input_t: x})
                        self.img_dump(enc, y, x, di.l, di.r)
