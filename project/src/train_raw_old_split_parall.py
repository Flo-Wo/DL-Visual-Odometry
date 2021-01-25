#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:06:46 2021

@author: florianwolf
"""
import time

import torch

from data_loader import generate_situation_splitting, generate_label_dict, path_labels, \
    DatasetOptFlo, generate_block_splitting, DatasetOptFlo1Frames

from network_trainer import setup_data_loader, train_network
from cnn.cnn_flow_only_with_pooling import CNNFlowOnlyWithPooling
from cnn.cnn_flow_only import CNNFlowOnly
from cnn.cnn_siamese_frames_flow import CnnSiamese
from network_user import process_video

data_size = 20399
block_size = 20399
train_eval_ratio = 0.8

MODEL_Conv = CNNFlowOnlyWithPooling(3, last_layer=True)
CRITERION_MSELoss = torch.nn.MSELoss()
OPTIMIZER_Adam_Conv = torch.optim.Adam(MODEL_Conv.parameters(), lr=1e-3)
SCHEDULER_RedLROnPlateau_Conv = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER_Adam_Conv, factor=0.9, patience=1)
'''
if __name__ == "__main__":
    time.sleep(30*5*5)

    #splitting = generate_block_splitting(data_size, train_eval_ratio, block_size)
    splitting = generate_situation_splitting(0.8, shuffle=True)

    labels = generate_label_dict(path_labels, data_size)

    train_tensor, validation_tensor = setup_data_loader(DatasetOptFlo, splitting, labels)

    train_network(train_tensor, validation_tensor, 25, "LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_MultLayer_SitSplitShuffle",
                  model=CNNFlowOnlyWithPooling(3, last_layer=True), criterion=CRITERION_MSELoss, optimizer=OPTIMIZER_Adam_Conv,
                  scheduler=SCHEDULER_RedLROnPlateau_Conv)
    
    MODEL_Conv = CNNFlowOnlyWithPooling(3, last_layer=True)
    CRITERION_MSELoss = torch.nn.MSELoss()
    OPTIMIZER_Adam_Conv = torch.optim.Adam(MODEL_Conv.parameters(), lr=1e-3)
    SCHEDULER_RedLROnPlateau_Conv = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER_Adam_Conv, factor=0.9, patience=1)
    
    splitting = generate_situation_splitting(0.8, shuffle=False)

    labels = generate_label_dict(path_labels, data_size)

    train_tensor, validation_tensor = setup_data_loader(DatasetOptFlo, splitting, labels)

    train_network(train_tensor, validation_tensor, 25, "LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_MultLayer_SitSplit",
                  model=MODEL_Conv, criterion=CRITERION_MSELoss, optimizer=OPTIMIZER_Adam_Conv,
                  scheduler=SCHEDULER_RedLROnPlateau_Conv)

    MODEL_Siamese = CnnSiamese(3, last_layer=True)
    CRITERION_MSELoss = torch.nn.MSELoss()
    OPTIMIZER_Adam_Siamese = torch.optim.Adam(MODEL_Siamese.parameters(), lr=1e-3)
    SCHEDULER_RedLROnPlateau_Siamese = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER_Adam_Siamese,
                                                                                  factor=0.9, patience=1)

    # splitting = generate_block_splitting(data_size, train_eval_ratio, block_size)
    splitting = generate_situation_splitting(0.8, shuffle=True)

    labels = generate_label_dict(path_labels, data_size)

    train_tensor, validation_tensor = setup_data_loader(DatasetOptFlo1Frames, splitting, labels)

    train_network(train_tensor, validation_tensor, 12, "LeakyReLU_SIAMESE_LastLayer_SitSplit_Shuffle",
                  model=MODEL_Siamese, criterion=CRITERION_MSELoss, optimizer=OPTIMIZER_Adam_Siamese,
                  scheduler=SCHEDULER_RedLROnPlateau_Siamese)
    
    MODEL_Siamese = CnnSiamese(3, last_layer=True)
    CRITERION_MSELoss = torch.nn.MSELoss()
    OPTIMIZER_Adam_Siamese = torch.optim.Adam(MODEL_Siamese.parameters(), lr=1e-3)
    SCHEDULER_RedLROnPlateau_Siamese = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER_Adam_Siamese,
                                                                                  factor=0.9, patience=1)
    
    splitting = generate_situation_splitting(0.8, shuffle=False)

    labels = generate_label_dict(path_labels, data_size)

    train_tensor, validation_tensor = setup_data_loader(DatasetOptFlo1Frames, splitting, labels)

    train_network(train_tensor, validation_tensor, 20, "LeakyReLU_SIAMESE_SitSplit_LastLayer",
                  model=MODEL_Siamese, criterion=CRITERION_MSELoss, optimizer=OPTIMIZER_Adam_Siamese,
                  scheduler=SCHEDULER_RedLROnPlateau_Siamese)
'''
video = "./data/raw/train.mp4"
'''
    process_video(video,
                  "./cnn/saved_models/NewSplitting/LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_MultLayer_SitSplit.pth",
                  "./cnn/saved_models/Videos/LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_MultLayer_SitSplitShuffle_train",
                  model=CNNFlowOnlyWithPooling(3, last_layer=True), dataset_class=DatasetOptFlo)

    process_video(video,
                  "./cnn/saved_models/NewSplitting/LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_MultLayer_SitSplit.pth",
                  "./cnn/saved_models/Videos/LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_MultLayer_SitSplit_train",
                  model=CNNFlowOnlyWithPooling(3, last_layer=True), dataset_class=DatasetOptFlo)
'''
process_video(video,
                  "./cnn/saved_models/NewSplitting/LeakyReLU_SIAMESE_LastLayer_SitSplit_Shuffle.pth",
                  "./cnn/saved_models/Videos/LeakyReLU_SIAMESE_LastLayer_SitSplit_Shuffle_train",
                  model=CnnSiamese(3, last_layer=True), dataset_class=DatasetOptFlo1Frames)

process_video(video,
                  "./cnn/saved_models/NewSplitting/LeakyReLU_SIAMESE_LastLayer_SitSplit_Shuffle.pth",
                  "./cnn/saved_models/Videos/LeakyReLU_SIAMESE_LastLayer_SitSplit_Shuffle_train",
                  model=CnnSiamese(3, last_layer=True), dataset_class=DatasetOptFlo1Frames)
