#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:08:41 2021

@author: florianwolf
"""

import logging, coloredlogs

from network_trainer import NetworkTrainer
from cnn.cnn_siamese_frames_flow import CnnSiamese
from data_loader import DatasetOptFlo
from cnn.cnn_flow_only_with_pooling import CNNFlowOnlyWithPooling
from cnn.cnn_flow_only import CNNFlowOnly



logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    path_labels = "./data/raw/train_label.txt"

    network_save_file = "leakyReLU8EpochsBatchNormMaxPooling"

    test_split_ratio = 0.8
    block_size = 3400

    dataLoader_params = {'batch_size': 64, 'shuffle': True}

    nwt = NetworkTrainer(20399, DatasetOptFlo, CNNFlowOnlyWithPooling)


    tr_tensor, eval_tensor = nwt.configure_data_loader(path_labels,
                            test_split_ratio, block_size, dataLoader_params)
    metadata = nwt.train_model(tr_tensor, eval_tensor, 3, 12, network_save_file)