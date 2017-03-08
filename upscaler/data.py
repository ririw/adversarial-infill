import luigi
from matplotlib.pyplot import imread
from glob import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data

images_location = '/home/riri/Datasets/flowers/jpg/'


class ImageReader(luigi.Task):
    box_size = luigi.IntParameter(default=32)
    remove_size = luigi.IntParameter(default=8)

    def output(self):
        return luigi.LocalTarget('./cache/image_dataset-{}.npz'.format(self.box_size))

    def run(self):
        images = glob(images_location + "*.jpg")
        slices = []
        for image in tqdm(images):
            img = imread(image).astype(np.float32) / 255
            for r in range(0, (img.shape[0] - self.box_size) // 4, self.box_size):
                for c in range(0, (img.shape[1] - self.box_size) // 2, self.box_size):
                    slice = img[None, r:r+self.box_size, c:c+self.box_size, :].copy()
                    slices.append(slice)

        slices = np.concatenate(slices, 0).transpose([0, 3, 1, 2])

        with open(self.output().path, 'wb') as f:
            np.save(f, slices)


    def load(self):
        return np.load(self.output().path, mmap_mode='r')

    def batch_iterator(self, batch_size):
        data = self.load()
        n = data.shape[0]
        permutation = np.random.permutation(n)
        for i in range(0, n-self.batch_size, self.batch_size):
            yield data[permutation[i:i+batch_size]]
