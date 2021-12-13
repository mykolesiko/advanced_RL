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


def init_layer(conv, mean, std, bias):
    conv.weight.data.normal_(mean, std)
    if bias:
        conv.bias.data.zero_()


class Down_module(nn.Module):
    def __init__(self, in_channel, out_channel, batch_norm=True, mean=0.0, std=0.02):
        super(Down_module, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(4, 4), stride=2, padding=1, bias=False)
        init_layer(self.conv, mean, std, False)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        output = F.leaky_relu(x, 0.3)
        return output


class Upper_module(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=False, batch_norm=True, mean=0.0, std=0.02):
        super(Upper_module, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(4, 4), stride=2, padding=1, bias=False)
        init_layer(self.conv_t, mean, std, False)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm_layer = nn.BatchNorm2d(out_channel)
        self.dropout = dropout
        if self.dropout:
            self.dropout_layer = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv_t(x)
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        if self.dropout:
            x = self.dropout_layer(x)
        output = F.relu(x)
        return output


class Generator(nn.Module):
    def __init__(self, mean=0.0, std=0.02):
        super(Generator, self).__init__()
        self.dlist = nn.ModuleList()
        batch_norm = [0, 1, 1, 1, 1, 1, 1, 0]
        d_input = [3, 64, 128, 256, 512, 512, 512, 512]
        d_output = d_input[1:] + [512]
        for i in range(len(d_input)):
            self.dlist.append(Down_module(d_input[i], d_output[i], batch_norm=batch_norm[i], mean=mean, std=std))

        self.ulist = nn.ModuleList()
        u_input = [512, 1024, 1024, 1024, 1024, 512, 256]
        u_output = [512, 512, 512, 512, 256, 128, 64]
        batch_norm = [1, 1, 1, 1, 1, 1, 1, 1]
        drops = [1, 1, 1, 0, 0, 0, 0, 0]
        for i in range(len(u_input)):
            self.ulist.append(
                Upper_module(u_input[i], u_output[i], drops[i], batch_norm=batch_norm[i], mean=mean, std=std))

        self.conv = nn.ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=2, padding=1)
        init_layer(self.conv, mean, std, True)

    def forward(self, x):

        skips = []
        for down in self.dlist:
            x = down(x)
            skips.append(x)

        skips = list(reversed(skips[:-1]))

        for up, skip in zip(self.ulist, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)

        output = torch.tanh(self.conv(x))
        return output


class Discriminator(nn.Module):
    def __init__(self, mean=0.0, std=0.02):
        super(Discriminator, self).__init__()
        self.dlist = nn.ModuleList()
        d_input = [6, 64, 128]
        d_output = d_input[1:] + [256]
        for i in range(len(d_input)):
            self.dlist.append(Down_module(d_input[i], d_output[i], batch_norm=(i != 0), mean=mean, std=std))
        self.conv1 = nn.Conv2d(256, 512, kernel_size=(4, 4), padding=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=(4, 4), padding=1)
        init_layer(self.conv1, mean, std, True)
        init_layer(self.conv2, mean, std, True)
        self.bn = nn.BatchNorm2d(512)

    def forward(self, x, masks):
        x = torch.cat([x, masks], 1)
        for block in self.dlist:
            x = block(x)
        # print(x.shape)

        x = self.conv1(x)
        x = self.bn(x)
        x = F.leaky_relu(x, 0.3)
        x = self.conv2(x)

        return torch.sigmoid(x)



def plot_images(images, masks, fakes):
    pil_images = []
    pil_masks = []
    pil_fakes = []
    batch_size = images.shape[0]
    for idx in range(batch_size):
        pil_images.append(transforms.ToPILImage()(images[idx]).convert("RGB"))
        pil_masks.append(transforms.ToPILImage()(masks[idx]).convert("RGB"))
        pil_fakes.append(transforms.ToPILImage()(fakes[idx]).convert("RGB"))
    fig = plt.figure(figsize=(20, 30))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, batch_size),  axes_pad=0.5 )
        
    images_to_show = pil_images + pil_masks + pil_fakes

    for ax, im in zip(grid, images_to_show):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()        