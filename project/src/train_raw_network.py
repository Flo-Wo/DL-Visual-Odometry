#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:06:46 2021

@author: florianwolf
"""

import torch
import numpy as np

from data_loader import generate_situation_splitting, generate_label_dict, path_labels, \
    DatasetOptFlo, generate_block_splitting, generate_hard_cut_off_splitting
    
from network_trainer import setup_data_loader, train_network
from cnn.cnn_flow_only_with_pooling import CNNFlowOnlyWithPooling
from cnn.cnn_flow_only import CNNFlowOnly

data_size = 20399
block_size = 100
train_eval_ratio = 0.8

standard_loader_params = {'batch_size': 64 , 'shuffle': False}

model_path = "./cnn/savedmodels/OriginalSplitting/LeakyReLU15EpochsBatchNormMaxPoolingWithDropOut.pth"

MODEL_Conv = CNNFlowOnlyWithPooling(3)
MODEL_Conv.load_state_dict(torch.load(model_path))
CRITERION_MSELoss = torch.nn.MSELoss()
OPTIMIZER_Adam_Conv = torch.optim.Adam(MODEL_Conv.parameters(), lr=1e-4)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(OPTIMIZER_Adam_Conv,
#                                                  milestones=[10],gamma=0.1,
#                                                  last_epoch=-1, verbose=True)

# lr_lambda = lambda epoch: 0.6
# scheduler = torch.optim.lr_scheduler.MultiplicativeLR(OPTIMIZER_Adam_Conv, lr_lambda, last_epoch=-1, verbose=True)
SCHEDULER_RedLROnPlateau_Conv = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER_Adam_Conv, factor=0.9, patience=1)

if __name__ == "__main__":
    # splitting = generate_block_splitting(data_size, train_eval_ratio, block_size)
    # splitting = generate_situation_splitting(0.8,augmentation=False)
    splitting = generate_hard_cut_off_splitting(0.8)

    labels = generate_label_dict(path_labels, data_size)

    train_tensor, validation_tensor = setup_data_loader(DatasetOptFlo, splitting, labels,params=standard_loader_params)

    velo_all = train_network(train_tensor, validation_tensor, 2, "LeakyReLU_Frames_Conv_100Split_20Epochs_BatchNorm_MaxPool_AUGMENTED",
                  model=MODEL_Conv, criterion=CRITERION_MSELoss, optimizer=OPTIMIZER_Adam_Conv,
                  scheduler=SCHEDULER_RedLROnPlateau_Conv)
    
    from network_user import plot_velocity_chart

    
    desired_label = "./data/raw/train_label.txt"
    own_label = "./results_test.txt"
    factor = 1.0
    own_array = np.genfromtxt(own_label)
    desired_array = np.genfromtxt(desired_label)
    plot_velocity_chart(own_label,kernel_size=20,factor=factor,kernel_color="red")
    plot_velocity_chart(desired_label,color="orange")

