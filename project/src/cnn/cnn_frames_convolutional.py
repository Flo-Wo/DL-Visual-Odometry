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


# use:
# https://arxiv.org/pdf/1709.08429.pdf
class CnnFramesConv(nn.Module):
    def __init__(self):
        # call super method to create instance of the network class
        super(CnnFramesConv, self).__init__()

        self.conv1 = conv_layer(6, 8, kernel_size=5, padding=3, stride=2)
        self.conv2 = conv_layer(8, 8, kernel_size=4, padding=2, stride=2)
        self.conv3 = conv_layer(8, 16, kernel_size=4, padding=2, stride=2)
        self.conv3_1 = conv_layer(16, 16, kernel_size=3, padding=1, stride=1)
        self.conv4 = conv_layer(16, 32, kernel_size=3, padding=1, stride=2)
        self.conv4_1 = conv_layer(32, 32, kernel_size=3, padding=1, stride=1)

        # now fully connected layers
        self.fc1 = fc_layer(32 * 8 * 11, 100)
        self.fc2 = fc_layer(100, 20)
        # no activation function in the last layer
        self.fc3 = nn.Linear(20, 1)
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
    def forward(self, x, y):
        # we use shared weight for feature extraction and add up the extracted
        # flow and image features, before transforming them into the fully
        # connected layers
        t = torch.cat((x, y), 1)

        t = self.conv1(t)
        t = self.conv2(t)
        t = self.conv3(t)
        t = self.conv3_1(t)
        t = self.conv4(t)
        t = self.conv4_1(t)

        # here we need a reshape, to pass the tensor into a fc
        # result: shape =  torch.Size([10, 64, 53, 73]), according to
        # https://discuss.pytorch.org/t/transition-from-conv2d-to-linear-layer-equations/93850/2
        # we need to reshape the output (flatten layer in matlab)
        t = t.view(-1, 32 * 8 * 11)

        # now we add the features together as proposed in
        # https://arxiv.org/pdf/2010.09925.pdf and in
        # https://www.mathworks.com/help/deeplearning/ug/train-a-siamese-network-to-compare-images.html,

        t = self.fc1(t)
        t = self.fc2(t)
        t = self.fc3(t)
        # remove all dimensions with size 1, so we get a tensor of the form
        # batchSize x 1 (in particular a scalar for each input image, which
        # is, what we want)
        return t.squeeze(1)

