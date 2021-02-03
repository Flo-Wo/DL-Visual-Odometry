#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Franz Herbst
"""
import torch

from data_loader import generate_block_splitting, generate_label_dict, path_labels, \
    generate_situation_splitting, DatasetFrames, generate_test_splitting, path_labels_test
from network_trainer import setup_data_loader, train_network, network_exists, load_splitting, save_splitting
from cnn.cnn_frames_convolutional import CnnFramesConv
# some constants
from project.src.network_user import plot_training_process

data_size = 20399
data_size_test = 13176
train_eval_ratio = 0.8
block_size = 100

MODEL_Conv = CnnFramesConv()
CRITERION_MSELoss = torch.nn.MSELoss()
OPTIMIZER_Adam_Conv = torch.optim.Adam(MODEL_Conv.parameters(), lr=1e-3)
SCHEDULER_RedLROnPlateau_Conv = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER_Adam_Conv, factor=0.9, patience=1)

net_name = "LeakyReLU_FramesCONV_SitSplit"

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

    train_tensor, validation_tensor, test_tensor = setup_data_loader(DatasetFrames, splitting, labels,
                                                                     test_ids=test_ids, test_labels=test_labels)

    n, log = train_network(train_tensor, validation_tensor, 48, net_name,
                  model=MODEL_Conv, criterion=CRITERION_MSELoss, optimizer=OPTIMIZER_Adam_Conv,
                  scheduler=SCHEDULER_RedLROnPlateau_Conv, test_tensor=test_tensor)

    plot_training_process(n)
