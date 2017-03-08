import logging
import os

import luigi
import matplotlib.pyplot as plt
import numpy as np
from neobunch import Bunch
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.autograd import Variable

import upscaler.data


class Autoencoding(luigi.Task):
    box_size = luigi.IntParameter(default=16)
    remove_size = luigi.IntParameter(default=4)
    batch_size = 128

    def requires(self):
        return upscaler.data.ImageReader(box_size=self.box_size, remove_size=self.remove_size)

    def torch_iterator(self, data):
        for x, y in self.batch_iterator(data):
            yield Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y))

    def batch_iterator(self, data):
        di = self.data_info(data)
        permutation = np.random.permutation(di.n)

        for i in range(0, di.n-self.batch_size, self.batch_size):
            batch_ix = permutation[i:i+self.batch_size]
            batch = data[batch_ix].copy()
            X = batch.copy()
            X[:, :, di.l:di.r, di.l:di.r] = 0
            y = batch[:, :, di.l:di.r, di.l:di.r]

            yield X, y

    def img_dump(self, enc, y, X, l, r):
        logging.warning('Writing images')
        ensure_image_dir()
        for i in tqdm(range(16)):
            xim = X.data[i].numpy()
            plt.imshow(xim.transpose([1,2,0]))
            plt.savefig('images/%02d-a.png' % i)
            plt.close()
            im = enc.data[i].numpy()
            expected = y.data[i].numpy()
            xim[:, l:r, l:r] = im
            plt.imshow(xim.transpose([1,2,0]))
            plt.savefig('images/%02d-b.png' % i)
            plt.close()
            xim[:, l:r, l:r] = expected
            plt.imshow(xim.transpose([1,2,0]))
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
        print(di)

        encoder = Encoder(self.remove_size)
        encoer_optimizer = optim.Adam(encoder.parameters())

        for x, y in self.torch_iterator(data):
            encoder.zero_grad()
            enc = encoder(x)

            loss = torch.nn.MSELoss()(enc, y)
            loss.backward()
            encoer_optimizer.step()

            print(loss.data.numpy())
            if np.random.uniform() < 0.1:
                self.img_dump(enc, y, x, di.l, di.r)


class Encoder(torch.nn.Module):
    def __init__(self, remove_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 3, 5)

    def forward(self, x):
        x = nn.LeakyReLU()(self.conv1(x))
        x = nn.LeakyReLU()(self.conv2(x))
        x = nn.LeakyReLU()(self.conv3(x))
        return x



def ensure_image_dir():
    try:
        os.makedirs('images')
    except OSError:
        pass

