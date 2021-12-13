import os
from pathlib import Path
import random

import numpy as np
import time
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tqdm.notebook import trange, tqdm

from matplotlib import pyplot as plt


class Dataset:
    def __init__(self, datapath='./facades/train', split='train'):  # , aligned=True):
        self.path = Path(datapath)
        self.load()
        self.toTensor = transforms.ToTensor()
        self.split = split
        self.test_resize = transforms.Resize((256, 256))
        self.train_resize = transforms.Resize((286, 286))

    def load(self):
        self.images = sorted([p for p in self.path.iterdir() if p.suffix == '.jpg'])
        random.shuffle(self.images)

    def transform(self, image, mask):
        image = self.train_resize(image)
        mask = self.train_resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(256, 256))
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        return image, mask

    # def normalize(self, input_image, real_image):
    #     input_image = np.array(input_image).astype(np.float32)
    #     real_image = np.array(real_image).astype(np.float32)
    #     #print(input_image)
    #     input_image = (input_image / 127.5) - 1
    #     real_image = (real_image / 127.5) - 1

    #     return input_image, real_image

    def __getitem__(self, idx):
        image_orig = Image.open(self.images[idx]).convert('RGB')
        w, h = image_orig.size
        if ((w != 512) or (h != 256)):
            print("!!!!!!!!!!!!!!!")
        area_image = (0, 0, 256, 256)
        area_mask = (256, 0, 512, 256)
        image = image_orig.crop(area_image)
        mask = image_orig.crop(area_mask)

        if self.split == 'train':
            image, mask = self.transform(image, mask)
        elif self.split == 'test':
            image = self.test_resize(image)
            mask = self.test_resize(mask)

        # image, mask = self.normalize(image, mask)
        image = self.toTensor(image)
        mask = self.toTensor(mask)

        return image, mask

    def __len__(self):
        return len(self.images)