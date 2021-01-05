#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 09:28:42 2020

@author: florianwolf
"""

import torch
import torch.nn as nn
import logging


# this way we can easily change the activation function for the whole network
def activation_func():
    return nn.ReLU()


# additional arguments default values are taken directly from pytorch
def conv_layer(num_in_channels, num_out_channels, kernel_size, stride=1, padding=0, dilation=1):
    return nn.Sequential(nn.Conv2d(num_in_channels, num_out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True), activation_func())


def fc_layer(num_input_channels, num_output_channels):
    return nn.Sequential(nn.Linear(num_input_channels, num_output_channels), activation_func())


# same architecture as the flow only cnn, but the forward function is different
class CnnSiamese(nn.Module):
    def __init__(self, num_input_channels):
        # call super method to create instance of the network class
        super(CnnSiamese, self).__init__()
        self.conv1f = conv_layer(num_input_channels, 24, kernel_size=5, stride=2)
        self.conv2f = conv_layer(24, 36, kernel_size=5, stride=2)
        self.conv3f = conv_layer(36, 36, kernel_size=5, stride=2)
        # randomly pick some channels/feature maps and zero them out
        self.dropf = nn.Dropout2d(p=0.5)
        self.conv4f = conv_layer(36, 36, kernel_size=3, stride=1)
        #self.conv5f = conv_layer(64, 64, kernel_size=3, stride=1)

        self.conv1o = conv_layer(num_input_channels, 24, kernel_size=5, stride=2)
        self.conv2o = conv_layer(24, 36, kernel_size=5, stride=2)
        self.conv3o = conv_layer(36, 36, kernel_size=5, stride=2)
        # randomly pick some channels/feature maps and zero them out
        self.dropo = nn.Dropout2d(p=0.5)
        self.conv4o = conv_layer(36, 36, kernel_size=3, stride=1)
        #self.conv5o = conv_layer(64, 64, kernel_size=3, stride=1)

        # now fully connected layers
        self.fc1 = fc_layer(64 * 6 * 13, 100)
        #self.fc1 = fc_layer(319488, 100)
        self.fc2 = fc_layer(100, 50)
        self.fc3 = fc_layer(50, 10)
        # no activation function in the last layer
        self.fc4 = nn.Linear(10, 1)
        # init weights and bias' for the convolution layer
        # for the linear layers, pytorch chooses a uniform distribution for
        # w and b, which should be fine
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # add nonlinearity to the layer, as we are using the relu function
                # this should also be the way how matlab chooses weights
                nn.init.kaiming_uniform_(layer.weight.data, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    # init bias with all zeros, as we usually did in the lecture
                    layer.bias.data.zero_()

        # print(self.modules)

    # implement forward function for the network, to take the flow and 
    # the image
    def forward(self, x, y, o):
        # we use shared weight for feature extraction and add up the extracted
        # flow and image features, before transforming them into the fully 
        # connected layers
        x = self.conv1f(x)
        y = self.conv1f(y)
        o = self.conv1o(o)

        x = self.conv2f(x)
        y = self.conv2f(y)
        o = self.conv2o(o)

        x = self.conv3f(x)
        y = self.conv3f(y)
        o = self.conv3o(o)

        x = self.dropf(x)
        y = self.dropf(y)
        o = self.dropo(o)

        x = self.conv4f(x)
        y = self.conv4f(y)
        o = self.conv4o(o)

        #x = self.conv5f(x)
        #y = self.conv5f(y)
        #o = self.conv5o(o)

        print(o.shape)
        exit(0)

        # here we need a reshape, to pass the tensor into a fc
        #logging.debug("shape = ", x.shape)
        #logging.debug("shape = ", y.shape)
        # result: shape =  torch.Size([10, 64, 53, 73]), according to
        # https://discuss.pytorch.org/t/transition-from-conv2d-to-linear-layer-equations/93850/2
        # we need to reshape the output (flatten layer in matlab)
        x = x.view(-1, 64 * 6 * 13)
        y = y.view(-1, 64 * 6 * 13)
        o = o.view(-1, 64 * 6 * 13)

        #x = x.view(-1, 319488)
        #y = y.view(-1, 319488)

        # now we add the features together as proposed in 
        # https://arxiv.org/pdf/2010.09925.pdf and in
        # https://www.mathworks.com/help/deeplearning/ug/train-a-siamese-network-to-compare-images.html,
        z = (x + y) * o

        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        z = self.fc4(z)
        # remove all dimensions with size 1, so we get a tensor of the form
        # batchSize x 1 (in particular a scalar for each input image, which
        # is, what we want)
        return (z.squeeze(1))
