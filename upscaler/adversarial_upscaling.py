import logging
import os
from pprint import pprint, pformat

import luigi
import matplotlib.pyplot as plt
import numpy as np
import shutil
from neobunch import Bunch
from tqdm import tqdm
from keras.utils import generic_utils
import torch
from torch import nn, optim
from torch.autograd import Variable

import upscaler.data


class AdversarialUpscaling(luigi.Task):
    box_size = luigi.IntParameter(default=16)
    remove_size = luigi.IntParameter(default=4)
    batch_size = 128
    n_rounds=10

    def requires(self):
        return upscaler.data.ImageReader(box_size=self.box_size, remove_size=self.remove_size)

    def torch_iterator(self, data, n_rounds):
        for x, y, missing in self.batch_iterator(data, n_rounds):
            yield Variable(torch.from_numpy(x)),\
                  Variable(torch.from_numpy(y)),\
                  Variable(torch.from_numpy(missing))

    def batch_iterator(self, data, n_rounds):
        di = self.data_info(data)

        for j in range(n_rounds):
            permutation = np.random.permutation(di.n)
            for i in range(0, di.n-self.batch_size, self.batch_size):
                batch_ix = permutation[i:i+self.batch_size]
                batch = data[batch_ix].copy()
                X = batch.copy()
                X[:, :, di.l:di.r, di.l:di.r] = 0
                missing_mask = np.zeros_like(X[:, 0:1, :, :])
                missing_mask[:, :, di.l:di.r, di.l:di.r] = 1
                y = batch[:, :, di.l:di.r, di.l:di.r]

                yield X, y, missing_mask

    def batch_len(self, data, n_rounds):
        di = self.data_info(data)
        return n_rounds * di.n // self.batch_size


    def img_dump(self, x_gen, x_true):
        x_gen = x_gen.data.numpy().transpose(1,2,0)
        x_true = x_true.data.numpy().transpose(1,2,0)

        for i in range(16):
            plt.imshow(x_gen[i].transpose(1, 2, 0))
            plt.savefig('./images/%d-a.jpg' % i)
            plt.close()
            plt.imshow(x_true[i].transpose(1, 2, 0))
            plt.savefig('./images/%d-b.jpg' % i)
            plt.close()


    def img_dump_one(self, x_gen, x_true, i, y):
        x_gen = x_gen[0].data.numpy()
        x_true = x_true[0].data.numpy()
        yv = y[0].data.numpy(0)
        x_all = np.concatenate([x_gen, x_true], 1)
        plt.imshow(x_all.transpose(1,2,0))
        plt.title('Classification: ' + ('real' if yv > 0.5 else 'gen') + str(yv))
        plt.savefig('./images/epoch-%04d.jpg' % i)
        plt.close()

    def data_info(self, data):
        n, c, w, h = data.shape
        center = w // 2
        l, r = center-self.remove_size, center+self.remove_size
        return Bunch(
            n=n, c=c, w=w, h=h, l=l, r=r, center=center
        )

    def run(self):
        shutil.rmtree('images', ignore_errors=True)
        os.mkdir('images')
        data = self.requires().load()
        di = self.data_info(data)

        generator = Generator(self.remove_size)
        checker = Checker()
        generator_opt = optim.Adam(generator.parameters())
        checker_opt = optim.Adam(checker.parameters())
        bar = tqdm(total=self.batch_len(data, self.n_rounds))
        i = 0
        loss = Variable(torch.from_numpy(np.zeros(1)))
        for x, y, missing in self.torch_iterator(data, 10):

            def make_mats():
                generator.zero_grad()
                checker.zero_grad()
                gen = generator(x, missing)

                top = x[:, :, :di.l, :]
                bottom = x[:, :, di.r:, :]
                left = x[:, :, di.l:di.r, :di.l]
                right = x[:, :, di.l:di.r, di.r:]

                mid_g = torch.cat([left, gen, right], 3)
                mid_y = torch.cat([left, y, right], 3)
                x_g = torch.cat([top, mid_g, bottom], 2)
                x_y = torch.cat([top, mid_y, bottom], 2)

                discrims_g = checker(x_g)# + Variable(torch.rand(x_g.size()), requires_grad=False)/10)
                discrims_y = checker(x_y)# + Variable(torch.rand(x_y.size()), requires_grad=False)/10)

                return x_g, x_y, discrims_g, discrims_y
            if ('adv' == 'adv' and np.random.uniform() > 0.5) or i < 1 or True:
                x_e, x_y, discrims_g, discrims_y = make_mats()
                generator_opt_loss = \
                    torch.nn.BCELoss()(discrims_g, Variable(torch.ones(self.batch_size))) + \
                    torch.nn.BCELoss()(discrims_y, Variable(torch.ones(self.batch_size)))

                generator_opt_loss.backward()
                generator_opt.step()

                x_e, x_y, discrims_g, discrims_y = make_mats()
                gen_opt_loss = \
                    torch.nn.BCELoss()(discrims_g, Variable(torch.zeros(self.batch_size))) + \
                    torch.nn.BCELoss()(discrims_y, Variable(torch.ones(self.batch_size)))

                gen_opt_loss.backward()
                checker_opt.step()

                bar.update(1)
                bar.set_description(str(
                    {'gen_loss': generator_opt_loss.data.numpy()[0],
                     'disc_loss': gen_opt_loss.data.numpy()[0],
                     'g_disc_mean': (discrims_g.data.numpy() > 0.5).mean(),
                     'y_disc_mean': (discrims_y.data.numpy() > 0.5).mean(),
                     'loss': loss.data.numpy()[0]
                     }))
            else:
                x_e, x_y, discrims_g, discrims_y = make_mats()
                loss = torch.nn.MSELoss()(x_e, x_y)
                loss.backward()
                generator_opt.step()

                bar.update(1)
                bar.set_description(str(
                    {'gen_loss': generator_opt_loss.data.numpy()[0],
                     'disc_loss': gen_opt_loss.data.numpy()[0],
                     'g_disc_mean': (discrims_g.data.numpy() > 0.5).mean(),
                     'y_disc_mean': (discrims_y.data.numpy() > 0.5).mean(),
                     'loss': loss.data.numpy()[0]
                     }))
            i += 1
            #if i % 10 == 1:
            #    self.img_dump(x_e, x_y)
            #    last_images = i
            self.img_dump_one(x_e, x_y, i, discrims_g)

        bar.close()

class Checker(torch.nn.Module):
    def __init__(self):
        super(Checker, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

    def forward(self, x):
        x = nn.MaxPool2d(2)(nn.LeakyReLU()(self.conv1(x)))
        x = nn.MaxPool2d(2)(nn.LeakyReLU()(self.conv2(x)))
        size = x.size()
        x = x.view(size[0], size[1]*size[2]*size[3])
        x = nn.LeakyReLU()(self.l1(x))
        x = nn.Sigmoid()(self.l2(x))
        return x


class Generator(torch.nn.Module):
    def __init__(self, remove_size):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, 5, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv4 = nn.Conv2d(128, 3, 3)

    def forward(self, x, missing):
        x = torch.cat([x, missing], 1)
        x = nn.LeakyReLU()(self.conv1(x))
        x = nn.LeakyReLU()(self.conv2(x))
        x = nn.LeakyReLU()(self.conv3(x))
        x = nn.LeakyReLU()(self.conv4(x))
        return x


if __name__ == '__main__':
    AdversarialUpscaling().run()