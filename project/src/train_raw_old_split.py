#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:06:46 2021

@author: florianwolf
"""

import torch

import numpy as np

from data_loader import generate_situation_splitting, generate_label_dict, path_labels, \
    DatasetOptFlo, generate_block_splitting, path_labels_test, generate_test_splitting

from network_trainer import setup_data_loader, train_network, network_exists, load_splitting, save_splitting, \
    network_folder
from cnn.cnn_flow_only_with_pooling import CNNFlowOnlyWithPooling
from cnn.cnn_flow_only import CNNFlowOnly
from matplotlib import pyplot as plt

from network_user import plot_training_process, plot_train_data_error, plot_test_data, plot_test_data_error
from project.src.network_user import plot_train_data

data_size = 20399
data_size_test = 13176
block_size = 20399
train_eval_ratio = 0.8

MODEL_Conv = CNNFlowOnlyWithPooling(3, last_layer=True)
CRITERION_MSELoss = torch.nn.MSELoss()
OPTIMIZER_Adam_Conv = torch.optim.Adam(MODEL_Conv.parameters(), lr=1e-3)
SCHEDULER_RedLROnPlateau_Conv = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER_Adam_Conv, factor=0.9, patience=1)

net_name = "LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_MultLayer_SitSplitShuffle"

if __name__ == "__main__":
    # splitting = generate_block_splitting(data_size, train_eval_ratio, block_size)
    if network_exists(net_name):
        splitting = load_splitting(net_name)
    else:
        splitting = generate_situation_splitting(0.9, shuffle=True)
        save_splitting(splitting, net_name)

    test_ids = generate_test_splitting(data_size_test)

    labels = generate_label_dict(path_labels, data_size)
    test_labels = generate_label_dict(path_labels_test, data_size_test)

    train_tensor, validation_tensor, test_tensor = setup_data_loader(DatasetOptFlo, splitting, labels,
                                                                     test_ids=test_ids, test_labels=test_labels)

    n, log = train_network(train_tensor, validation_tensor, 15, net_name, model=MODEL_Conv,
                           criterion=CRITERION_MSELoss, optimizer=OPTIMIZER_Adam_Conv,
                           scheduler=SCHEDULER_RedLROnPlateau_Conv, test_tensor=test_tensor)

    plot_training_process(n)
