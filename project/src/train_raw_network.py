#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:06:46 2021

@author: florianwolf
"""

import torch

from data_loader import generate_situation_splitting, generate_label_dict, path_labels, \
    DatasetOptFlo
    
from network_trainer import setup_data_loader, train_network
from cnn.cnn_flow_only_with_pooling import CNNFlowOnlyWithPooling

data_size = 20399

MODEL_Conv = CNNFlowOnlyWithPooling(3)
CRITERION_MSELoss = torch.nn.MSELoss()
OPTIMIZER_Adam_Conv = torch.optim.Adam(MODEL_Conv.parameters(), lr=1e-4)
SCHEDULER_RedLROnPlateau_Conv = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER_Adam_Conv, factor=0.9, patience=1)

if __name__ == "__main__":
    #splitting = generate_block_splitting(data_size, train_eval_ratio, block_size)
    splitting = generate_situation_splitting(0.8)

    labels = generate_label_dict(path_labels, data_size)

    train_tensor, validation_tensor = setup_data_loader(DatasetOptFlo, splitting, labels)

    train_network(train_tensor, validation_tensor, 10, "LeakyReLU_Frames_Conv_SitSplit_10Epochs_BatchNorm_MaxPool_17-01-2021",
                  model=MODEL_Conv, criterion=CRITERION_MSELoss, optimizer=OPTIMIZER_Adam_Conv,
                  scheduler=SCHEDULER_RedLROnPlateau_Conv)

