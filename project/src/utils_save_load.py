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

    

def load_images():
    # do it like 
    # https://www.geeksforgeeks.org/opencv-the-gunnar-farneback-optical-flow/
    video = cv2.VideoCapture("data/raw/train.mp4")
    success,prev_frame = video.read()
    while success:
        success, curr_frame = video.read()
        yield prev_frame, curr_frame
        prev_frame = curr_frame
    video.release()
    cv2.destroyAllWindows()
    
def sample_down_half(frame):
    return(cv2.resize(frame,(320,220)))


def calc_of(curr_frame, prev_frame):
    curr_frame, prev_frame = sample_down_half(curr_frame[:-40,:,:]), sample_down_half(prev_frame[:-40,:,:]) #
    # Create mask 
    hsv_mask = np.zeros_like(prev_frame) 
    # Make image saturation to a maximum value 
    hsv_mask[:,:, 1] = 255
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Optical flow is now calculated 
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                        None,\
                                        pyr_scale=0.5, levels=3, winsize=6,\
                                        iterations=3, poly_n=5, poly_sigma=1.1,\
                                        flags=0)
    # Compute magnitude and angle of 2D vector 
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
    # Set image hue value according to the angle of optical flow 
    hsv_mask[:,:, 0] = ang * (180 / np.pi / 2)
    # Set value as per the normalized magnitude of optical flow 
    hsv_mask[:,:, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) 
    # Convert to rgb 
    rgb_image = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2RGB) 
    return(rgb_image)
    

def generate_and_save_torch_dataset(label_path,save_path):
    # load images, transform to tensors and add the label
    for i, (prev_frame, curr_frame) in enumerate(load_images()):
        if i%100 == 0:
            print(i)
        #filepath = flow_dataset_path + "/{:05d}_of.png".format(i)
        rgb_flow = calc_of(curr_frame, prev_frame)
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
    # we cant compute the optical flow for the first frame
    dataset_all = TensorDataset(flow_stack,\
                                 torch.from_numpy(labels[1:]))
    save_path_name = save_path + "optical_flow_with_labels_tensor"
    torch.save(dataset_all,save_path_name)
    return(dataset_all)




if __name__ == "__main__":
    dataset = generate_and_save_torch_dataset("./data/raw/train_label.txt","./data/tensorData/")
    #load_data("./data/optical_flow", test_split_ratio=0.7, batch_size=40)
# image = cv2.imread("./data/frames/frame999.png")
# image2 = cv2.imread("./data/frames/frame1000.png")
# res = calc_of(image,image2)

# cv2.imwrite("data/of7_without.png",res)