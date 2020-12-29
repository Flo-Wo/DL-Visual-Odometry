#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:36:35 2020

@author: florianwolf
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import datasets,transforms
import cv2
import numpy as np


# we use this file to laod the data into a torch dataloader, to efficiently
# store the data set and split it into train and test data


# we use https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# to load the data
def load_data(dataset_path, test_split_ratio, batch_size_own):
    data = torch.load(dataset_path)
    data_size = len(data)
    # now do the splitting
    all_indices = list(range(data_size))
    split_index = int(np.floor(data_size*test_split_ratio))
    train_indices = all_indices[:split_index]
    test_indices = all_indices[split_index:]
    train_data = Subset(data, train_indices)
    test_data = Subset(data, test_indices)
    # now create a torch data loader, to save the data efficiently
    train_loader = DataLoader(train_data,batch_size=batch_size_own)
    test_loader = DataLoader(test_data,batch_size=batch_size_own)
    return(train_loader,test_loader)

    
def generate_and_save_torch_dataset(flow_dataset_path, label_path,save_path):
    # load images, transform to tensors and add the label
    for i in range(1,100):
        filepath = flow_dataset_path + "/of_frame{}.png".format(i)
        rgb_flow = cv2.imread(filepath)
        #print(rgb_flow)
        # transform image to a tensor and concat them
        rgb_flow_tensor = transforms.ToTensor()(rgb_flow).unsqueeze(0)
        #print(rgb_flow_tensor.shape)
        if i == 1:
            flow_stack = rgb_flow_tensor
        else:
            flow_stack = torch.cat([flow_stack,rgb_flow_tensor])
    # load txt file with labels
    labels = np.loadtxt(label_path)
    print(flow_stack.shape)
    dataset_all = TensorDataset(flow_stack,\
                                 torch.from_numpy(labels[:99]))
    save_path_name = save_path + "tensor_of_with_labels"
    torch.save(dataset_all,save_path_name)
    return(dataset_all)




if __name__ == "__main__":
    dataset = generate_and_save_torch_dataset("./data/optical_flow", "./data/raw/train_label.txt","./data/tensorData/")
    #load_data("./data/optical_flow", test_split_ratio=0.7, batch_size=40)