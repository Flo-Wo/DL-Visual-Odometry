#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Franz Herbst
"""

from data_loader import generate_block_splitting, generate_label_dict, DatasetOptFlo1Frames, path_labels
from network_trainer import setup_data_loader, train_network
from network_trainer import MODEL_Siamese, CRITERION_MSELoss, OPTIMIZER_Adam_Siamese, SCHEDULER_RedLROnPlateau_Siamese

# some constants
data_size = 20399
train_eval_ratio = 0.8
block_size = 100

if __name__ == "__main__":
    splitting = generate_block_splitting(data_size, train_eval_ratio, block_size)
    labels = generate_label_dict(path_labels, data_size)

    train_tensor, validation_tensor = setup_data_loader(DatasetOptFlo1Frames, splitting, labels)

    train_network(train_tensor, validation_tensor, 20, "LeakyReLU_SIAMESE",
                  model=MODEL_Siamese, criterion=CRITERION_MSELoss, optimizer=OPTIMIZER_Adam_Siamese,
                  scheduler=SCHEDULER_RedLROnPlateau_Siamese)
