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
import data_loader
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

# load data 
train_loader, test_loader = data_loader.get_loaders(64, 4)

# Model definition
class LSTMVAENETWORK(nn.Module):
    def __init__(self, input_size, num_units_lstm):
        super(LSTMVAENETWORK, self).__init__()

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
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMVAENETWORK(img_size, num_units_lstm).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, _ = data
        
        inputs.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        recon_batch, mu, logvar =  model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
