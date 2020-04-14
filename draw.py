# author: Vipul Vaibhaw

"""
Simple Implementation of https://arxiv.org/pdf/1502.04623.pdf in PyTorch

Example Usage: 
    python3 draw.py 

"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import torchvision 
import torchvision.transforms as transforms

import os, sys

torch.manual_seed(1) 

# Defining some constants
width, height, depth = (28, 28, 1)
img_size = width*height # canvas
batch_size = 100 
epochs = 10000
lr = 1e-3
eps = 1e-8

glimpses = 64
read_glimpse = 2 # table 3 
read_glimpse_classification = 12 # table 3
write_glimpse = 5 # table 3
z_size = 100 # table 3
num_units_lstm = 256 # table 3

# Model definition
class LSTMNETWORK(nn.Module):
    def __init__(self, input_size, num_units_lstm):
        super(LSTMNETWORK, self).__init__()

        # take the input
        self.encoderrnn = nn.LSTM(input_size, num_units_lstm)

        # a few linear layers needed for reparameterize trick
        self.mu = nn.Linear(num_units_lstm, z_size)
        self.logvar = nn.Linear(num_units_lstm, z_size)

        # take the sampled output and regenerate the image
        self.decoderrnn = nn.LSTM(z_size, input_size)

    def encoder(self, x):
        x = self.encoderrnn(x)
        return nn.ReLU(self.mu(x)), nn.ReLU(self.logvar(x))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + eps*std

    def decode(self, z):
        z = z.view(-1,num_units_lstm)
        return self.decoderrnn(z)

    def forward(self, x):
        x = x.view(-1, img_size)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)


