import random
from scipy.misc import imread
import numpy as np
import os
from glob import glob

def load_dataset(dir, ext='png'):
    paths = glob(os.path.join(dir, "*."+ext))
    return DataSet(paths)

class DataSet(object):

    def __init__(self, paths):
        self.paths = paths
        self.index_in_epoch = 0
        self.num_examples = len(paths)
        self.epochs_completed = 0

    def sample_img(self):
        return imread(self.paths[0], mode='RGBA')

    def next_batch(batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            self.epochs_completed += 1
            random.shuffle(self.paths)
            start = 0
            self.index_in_epoch = batch_size
        end = self.index_in_epoch
        imgs = [imread(path, mode='RGBA') for path in self.paths[start:end]]
        return np.array(imgs).astype(np.float32)
