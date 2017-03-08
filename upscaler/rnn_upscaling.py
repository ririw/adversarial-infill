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


class RNNUpscaling(luigi.Task):
    box_size = luigi.IntParameter(default=8)
    remove_size = luigi.IntParameter(default=2)
    batch_size = 128

    def requires(self):
        return upscaler.data.ImageReader(box_size=self.box_size, remove_size=self.remove_size)

    def torch_iterator(self, data):
        for x, y in self.batch_iterator(data):
            yield Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y))


    def data_info(self, data):
        n, c, w, h = data.shape
        center = w // 2
        l, r = center-self.remove_size, center+self.remove_size
        return Bunch(n=n, c=c, w=w, h=h, l=l, r=r, center=center, rs=self.remove_size)

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

    def run(self):
        data = self.requires().load()
        di = self.data_info(data)
        prnn = PixelRNN(di)
        opt = optim.Adam(prnn.parameters())

        for x, y in self.torch_iterator(data):
            prnn.zero_grad()
            enc = prnn(x)
            loss = torch.nn.MSELoss()(enc, y)
            loss.backward()
            opt.step()
            print(loss)



class PixelRNN(torch.nn.Module):
    def __init__(self, data_info):
        super(PixelRNN, self).__init__()
        self.rnn1 = torch.nn.LSTMCell(3 * data_info.rs, 64)
        #self.rnn2 = torch.nn.LSTMCell(64, 3)
        self.l1 = torch.nn.Linear(64, 3)
        self.data_info = data_info

    def forward(self, x):
        self.rnn1.reset_parameters()

        di = self.data_info
        cx = Variable(torch.zeros(x.size()[0], 64), requires_grad=False)
        hx = Variable(torch.zeros(x.size()[0], 64), requires_grad=False)

        for r in range(di.l, di.r):
            for c in range(di.l, di.r):
                feed = x[:, :, r, c - di.rs:c].clone()
                feed_size = feed.size()
                feed = feed.view(feed_size[0], feed_size[1] * feed_size[2])
                hx, cx = self.rnn1.forward(feed, (hx, cx))
                v = torch.nn.LeakyReLU()(self.l1(hx))
                x = x.clone()
                x[:, :, r, c] = v

        return x[:, :, di.l:di.r, di.l:di.r].clone()

def ensure_image_dir():
    try:
        os.makedirs('images')
    except OSError:
        pass

