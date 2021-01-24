#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 10:44:11 2021

@author: florianwolf
"""
import numpy as np


import torch

from data_loader import generate_situation_splitting, generate_label_dict, path_labels, \
    DatasetOptFlo, generate_block_splitting
    
from network_trainer import setup_data_loader, train_network
from cnn.cnn_flow_only_with_pooling import CNNFlowOnlyWithPooling
from cnn.cnn_flow_only import CNNFlowOnly
from data_loader import DatasetOptFlo

import matplotlib.pyplot as plt

from network_user import process_video, plot_velocity_chart

video_path = "./data/raw/train.mp4"
model_path = "./cnn/savedmodels/OriginalSplitting/LeakyReLU15EpochsBatchNormMaxPoolingWithDropOut.pth"
save_path = "./results_test_load_tensor"
model = CNNFlowOnlyWithPooling(3)

own_label = save_path + ".txt"
desired_label = "./data/raw/train_label.txt"
own_label_old = "./results_test.txt"

own_array = np.genfromtxt(own_label)
desired_array = np.genfromtxt(desired_label)

if __name__ == "__main__":
    # #process_video(video_path, model_path, save_path, model=model,dataset_class=DatasetOptFlo)
    # factor = 1.0
    # plot_velocity_chart(own_label,kernel_size=20,factor=factor,kernel_color="red")
    # plot_velocity_chart(desired_label,color="orange")
    # #plot_velocity_chart(own_label_old,kernel_size=20,factor=factor,kernel_color="blue")
    # variance = np.var(factor*own_array - desired_array[1:])
    # mean = np.mean(factor*own_array - desired_array[1:])
    # print(variance)
    # print(mean)
    # first = np.loadtxt("velo_all_0")
    # second = np.loadtxt("velo_all_1")
    # desired = np.loadtxt(desired_label)
    
    # #plt.plot(first,"b",label="first epoch")
    # kernel_size=14
    # kernel = np.ones(10)/kernel_size
    # first_conv = np.convolve(first, kernel, mode='same')
    # plt.plot(first, "gray")
    # plt.plot(first_conv,"b-",label="second epoch")
    # plt.plot(desired,"r--",label="real value")
    # plt.legend(loc="upper right")
    
    plot_velocity_chart("./velo_all_0_no_grad_without_shuffle",kernel_size=10,kernel_color="red",factor=1.0)
    plot_velocity_chart(desired_label,color="blue")
    
    velo = np.loadtxt("./velo_all_0_no_grad")
    desired = np.loadtxt(desired_label)
    mean = np.mean(velo-desired[1:])
    var = np.var(velo-desired[1:])
    print(mean)
    print(var)
    
    