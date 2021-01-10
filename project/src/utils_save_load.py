#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:36:35 2020

@author: florianwolf
"""

import torch
#from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm


# we use this file to laod the data into a torch dataloader, to efficiently
# store the data set and split it into train and test data

# override the Dataset class of pytorch to efficiently run the code
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
# using this technique

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization with two dicts'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/tensorData/of/' + str(ID) + '.pt')
        y = self.labels[ID]

        return X, y

class Dataset_of_frames(torch.utils.data.Dataset):
    # class to return mini batches to the siamese network
    # here X1 is the optical flow frame and X2 is the downsampled, but 
    # raw frame, y is the label
    # we can decide here, how we want to pass this into the network
    # (siamese or linear combination)
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels):
          'Initialization with two dicts'
          self.labels = labels
          self.list_IDs = list_IDs
    
    def __len__(self):
          'Denotes the total number of samples'
          return len(self.list_IDs)
    
    def __getitem__(self, index):
          'Generates one sample of data'
          # Select sample
          ID = self.list_IDs[index]
    
          # Load data and get label
          X1 = torch.load('data/tensorData/of/' + "{:05d}.pt".format(ID))
          X2 = torch.load("data/tensorData/frames/" + "{:05d}.pt".format(ID))
          y = self.labels[ID]
    
          return X1, X2, y

#### SAVE THE IMAGES AS TENSORS ####

def load_images_single():
    # only load the single images, to sample them down and save them as tensors
    # do it like 
    # https://www.geeksforgeeks.org/opencv-the-gunnar-farneback-optical-flow/
    video = cv2.VideoCapture("data/raw/train.mp4")
    success,prev_frame = video.read()
    while success:
        success, curr_frame = video.read()
        if success == False:
            break
        yield curr_frame
    video.release()
    cv2.destroyAllWindows()
    
def save_frames_as_tensors(save_path):
    print("saving single frames!")
    # load images, transform to tensors and add the label
    for i, frame in enumerate(tqdm(load_images_single(), "Frames as Tensors")):
        #if i%100 == 0:
        #    print(i)
        frame = sample_down_half(frame[:-60,:,:])
        frame = sample_down_half_second(frame)
        frame = transforms.ToTensor()(frame)#.unsqueeze(0)
        #print(frame.shape)
        torch.save(frame, save_path + "{:05d}.pt".format(i+1))

#### CALCULATION AND SAVING OF THE OPTICAL FLOW #####

def generate_flow_all(save_path):
    print("saving of tensors!")
    # load images, transform to tensors and add the label
    for i, (prev_frame, curr_frame) in enumerate(tqdm(load_images(), "Flow as Tensors")):
        #if i%100 == 0:
        #    print(i)
        #filepath = flow_dataset_path + "/{:05d}_of.png".format(i)
        rgb_flow = calc_of(curr_frame, prev_frame)
        #print(rgb_flow)
        #cv2.imwrite("./data/opticalflow/" + str(i) + "_of.png", rgb_flow)
        # transform image to a tensor and concat them
        rgb_flow_tensor = transforms.ToTensor()(rgb_flow)#.unsqueeze(0)
        #print(rgb_flow_tensor.shape)
        torch.save(rgb_flow_tensor,save_path + "{:05d}.pt".format(i+1))

def generate_train_eval_dict(data_size, test_split_ratio):
    # we have 20399 images and of frames, we split them and create a dict
    data_size= 20399
    split_index = int(np.floor(data_size*test_split_ratio))
    all_indices = list(range(1,data_size+1))
    train_indices = ["{:05d}".format(x) for x in all_indices[:split_index]]
    test_indices = ["{:05d}".format(x) for x in all_indices[split_index:]]
    partition = {'train' : train_indices,\
                 'validation' : test_indices}
    return(partition)

def generate_train_eval_dict_new_splitting(data_size, test_split_ratio):
    # we have 20399 images and of frames, we split them and create a dict
    ratio_percent = int(test_split_ratio * 100)
    all_indices = list(range(1, data_size + 1))

    train_indices = []
    test_indices = []

    for index in range(0,int(data_size/100)+1):
        if index == int(data_size/100)+1:
            train_indices.extend(all_indices[100*index:100*index + ratio_percent])
            test_indices.extend(all_indices[100*index + ratio_percent:])
        else:
            train_indices.extend(all_indices[100*index:100*index + ratio_percent])
            test_indices.extend(all_indices[100*index + ratio_percent:(index+1)*100])

    train_indices = ["{:05d}".format(x) for x in train_indices]
    train_indices = ["{:05d}".format(x) for x in test_indices]
    partition = {'train' : train_indices,\
                 'validation' : test_indices}
    return(partition)


def generate_label_dict(label_path,data_size):
    # these are all labels in a txt, we want to write them into a dict and
    # ignore the first value
    labels_np_array = np.loadtxt(label_path)
    labels = {}
    for index in range(1,data_size+1):
        #labels["{:05d}".format(index)] = labels_np_array[index]
        labels[index] = labels_np_array[index]
    return(labels)


def load_images():
    # do it like 
    # https://www.geeksforgeeks.org/opencv-the-gunnar-farneback-optical-flow/
    video = cv2.VideoCapture("data/raw/train.mp4")
    success,prev_frame = video.read()
    while success:
        success, curr_frame = video.read()
        if success == False:
            break
        yield prev_frame, curr_frame
        prev_frame = curr_frame
    video.release()
    cv2.destroyAllWindows()
    
def sample_down_half(frame):
    # sample to half size using bilinear interpolation
    return(cv2.resize(frame,(320,210)))

def sample_down_half_second(frame):
    # sample the image again down for half the size, after the of is calculated
    return(cv2.resize(frame,(160,105)))

def calc_of(curr_frame, prev_frame):
    # do it like 
    # https://www.geeksforgeeks.org/opencv-the-gunnar-farneback-optical-flow/
    curr_frame, prev_frame = sample_down_half(curr_frame[:-60,:,:]), sample_down_half(prev_frame[:-60,:,:])
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
    rgb_image = sample_down_half_second(rgb_image)
    return(rgb_image)





# #############################################################
# Create Overlay and create Video
# #############################################################

def overlay_speed_error_on_video(video_path, predicted_velocity_path,frame_limit=None,\
                                 velocity_path=None):
    # source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#saving-a-video
    
    if velocity_path is not None:
        label = True
        actual_velocity = np.loadtxt(velocity_path)
    else:
        label = False
    predicted_velocity = np.loadtxt(predicted_velocity_path)
    # load the original video
    video_original = cv2.VideoCapture(video_path)
    video_label = cv2.VideoWriter("./data/demos/results_siamese_test.mp4",\
                                  0x7634706d, 20, (640, 480))
    
    if frame_limit is None:
        frame_limit = len(predicted_velocity)
    
    success,frame = video_original.read()
    count = 0
    while success and count < frame_limit-1:
        if count >= 1:
            if label:
                frame_labeled = put_velo_error_on_frame(frame, predicted_velocity[count],\
                                                        velocity=actual_velocity[count])
            else:
                frame_labeled = put_velo_error_on_frame(frame, predicted_velocity[count])
            video_label.write(frame_labeled)
            success,frame = video_original.read()
            if count % 100 == 0:
                print('Frame: ', count)
        count += 1
    video_label.release()
    video_original.release()
    cv2.destroyAllWindows()

def put_velo_error_on_frame(frame, prediction, **kwargs):
    # set some important constants
    font = cv2.FONT_HERSHEY_SIMPLEX
    velo_color = (25, 255, 25)
    pred_color = (255, 25, 25)
    err_color = (25,25,255)
    fontScale = 1.1
    thickness = 2
    upper_offset = 40
    line_offset = 30
    right_offset = 8
    
    
    # check, whether the real velocity is given
    if "velocity" in kwargs:
        velocity = kwargs["velocity"]
        error = np.abs(velocity-prediction)
        velo_position = (right_offset,upper_offset)
        pred_position = (right_offset,upper_offset + line_offset)
        err_position = (right_offset,upper_offset + 2*line_offset)
        pred = "pred (m/s): " + "{:2.3f}".format(prediction)
        velo = "speed (m/s): " + "{:2.3f}".format(velocity)
        
        frame_labeled = cv2.putText(frame,velo,velo_position,font,fontScale,\
                                 velo_color,thickness)
        err = "abs. error: " + "{:2.3f}".format(error)
        frame_labeled = cv2.putText(frame_labeled,err,err_position,font,fontScale,\
                                 err_color,thickness)
        
        frame_labeled = cv2.putText(frame,pred,pred_position,font,fontScale,\
                                    pred_color,thickness)
    else:
        pred_position = (right_offset,upper_offset)
        pred = "pred (m/s): " + "{:2.3f}".format(prediction)
        frame_labeled = cv2.putText(frame,pred,pred_position,font,fontScale,\
                                    pred_color,thickness)
    return(frame_labeled)



    

if __name__ == "__main__":
    # i1 = cv2.imread("./data/frames/frame1.png")
    # i2 = cv2.imread("./data/frames/frame2.png")
    # i2_cut_down = sample_down_half(i2[:-60,:,:])
    # flow_field = calc_of(i1, i2)
    # cv2.imwrite("../report/imgs/frame2_original.png",i2)
    # cv2.imwrite("../report/imgs/frame2_cut_sampled.png",i2_cut_down)
    # cv2.imwrite("../report/imgs/frame2_flow_field.png",flow_field)
    # generate_flow_all("./data/tensorData/of/")
    # save_frames_as_tensors("./data/tensorData/frames/")
    frame = cv2.imread("./data/frames/frame1.png")
    frame_label = put_velo_error_on_frame(frame,20,velocity=20)
    cv2.imwrite("./test.png",frame_label)
    # overlay_speed_error_on_video("./data/raw/train.mp4",\
    #                              "./data/predicts/train_predicts_siamese.txt",\
    #                             20399,\
    #                             "./data/raw/train_label.txt")
    # overlay_speed_error_on_video("./data/raw/test.mp4",\
    #                              "./data/predicts/test_predicts_siamese.txt")
    pass

