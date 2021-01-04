#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03.01.2021

@author: franzherbst
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import cv2
import numpy as np
from tqdm import tqdm

# we use this file to laod the data into a torch dataloader, to efficiently
# store the data set and split it into train and test data

# override the Dataset class of pytorch to efficiently run the code
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
# using this technique

# #############################################################
# IMPORTANT CONSTANTS
# #############################################################

path_tensor_opt_fl = "./data/tensorData/of/"
path_tensor_frames = "./data/tensorData/frames/"
path_image_opt_fl = "data/opticalFlow/"
path_image_frames = "data/frames/"
path_raw_video = "data/raw/train.mp4"

picture_bottom_offset = 60
picture_opt_fl_size = (320, 210)
picture_final_size = (160, 105)


# #############################################################
# DATASET CLASSES
# #############################################################

class DatasetOptFlo(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_ids, labels):
        'Initialization with two dicts'
        self.list_IDs = list_ids
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        element_id = self.list_IDs[index]

        # Load data and get label
        x = torch.load(path_tensor_opt_fl + "{:05d}.pt".format(element_id))
        y = (self.labels[element_id] + self.labels[element_id - 1]) / 2

        return x, y


class DatasetFrames(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_ids, labels):
        'Initialization with two dicts'
        self.list_IDs = list_ids
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X1 = torch.load(path_tensor_frames + "{:05d}.pt".format(ID - 1))
        X2 = torch.load(path_tensor_frames + "{:05d}.pt".format(ID))
        y = (self.labels[ID] + self.labels[ID - 1]) / 2

        return X1, X2, y


class DatasetOptFlo1Frames(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_ids, labels):
        'Initialization with two dicts'
        self.list_IDs = list_ids
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        F = torch.load(path_tensor_frames + "{:05d}.pt".format(ID))
        X = torch.load(path_tensor_opt_fl + "{:05d}.pt".format(ID))
        y = (self.labels[ID] + self.labels[ID - 1]) / 2

        return F, X, y


class DatasetOptFlo2Frames(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_ids, labels):
        'Initialization with two dicts'
        self.list_IDs = list_ids
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        F1 = torch.load(path_tensor_frames + "{:05d}.pt".format(ID - 1))
        F2 = torch.load(path_tensor_frames + "{:05d}.pt".format(ID))
        X = torch.load(path_tensor_opt_fl + "{:05d}.pt".format(ID))
        y = (self.labels[ID] + self.labels[ID - 1]) / 2

        return F1, F2, X, y


# #############################################################
# IMAGE LOADERS
# #############################################################

def load_single_images(video_path):
    """
    Loading the single images out of the video to sample them down and save them as tensors
    Source: https://www.geeksforgeeks.org/opencv-the-gunnar-farneback-optical-flow/
    """
    video = cv2.VideoCapture(video_path)

    success, prev_frame = video.read()
    while success:
        yield curr_frame
        success, curr_frame = video.read()

    video.release()
    cv2.destroyAllWindows()


def load_double_images(video_path):
    """
    Loading two images out of the video to sample them down and calculate the optical flow
    Source: https://www.geeksforgeeks.org/opencv-the-gunnar-farneback-optical-flow/
    """
    video = cv2.VideoCapture(video_path)

    success, prev_frame = video.read()
    if success:
        success, curr_frame = video.read()

        while success:
            yield prev_frame, curr_frame
            prev_frame = curr_frame
            success, curr_frame = video.read()

    video.release()
    cv2.destroyAllWindows()


# #############################################################
# SAMPLE DOWN IMAGES
# #############################################################

def sample_down(frame, size):
    """Sample down the frame to special height"""
    return cv2.resize(frame, size)


def cut_bottom(frame, height):
    """Cuts off the bottom of the image"""
    return frame[:-height, :, :]


# #############################################################
# SAVE TENSORS
# #############################################################

def save_both(save_path_frames, save_path_of, video_path,
              save_as_png=False, save_png_fr=path_image_frames, save_png_of=path_image_opt_fl):
    """Iterate through video and save images and optical flow as tensors"""
    for i, (prev_frame, curr_frame) in enumerate(tqdm(load_double_images(video_path), "Save Flow and Frame Tensors")):
        curr_frame = sample_down(cut_bottom(curr_frame, picture_bottom_offset), picture_opt_fl_size)
        prev_frame = sample_down(cut_bottom(prev_frame, picture_bottom_offset), picture_opt_fl_size)

        # SAVE FRAME

        if i == 0:
            frame = sample_down(prev_frame, picture_final_size)
            if save_as_png:
                cv2.imwrite(save_png_fr + "{:05d}.png".format(i), frame)

            frame = transforms.ToTensor()(frame)  # .unsqueeze(0)
            # print(frame.shape)
            torch.save(frame, save_path_frames + "{:05d}.pt".format(i))

        frame = sample_down(curr_frame, picture_final_size)
        if save_as_png:
            cv2.imwrite(save_png_fr + "{:05d}.png".format(i), frame)

        frame = transforms.ToTensor()(frame)  # .unsqueeze(0)
        # print(frame.shape)
        torch.save(frame, save_path_frames + "{:05d}.pt".format(i))

        # SAVE FLOW

        rgb_flow = calculate_opt_flow(curr_frame, prev_frame)
        # print(rgb_flow)
        if save_as_png:
            cv2.imwrite(save_png_of + "{:05d}.png".format(i + 1), rgb_flow)
        # transform image to a tensor and concat them
        rgb_flow_tensor = transforms.ToTensor()(rgb_flow)  # .unsqueeze(0)
        # print(rgb_flow_tensor.shape)
        torch.save(rgb_flow_tensor, save_path_of + "{:05d}.pt".format(i + 1))


def save_frames_as_tensors(save_path, video_path, save_as_png=False, save_png_path=path_image_frames):
    """load images, transform to tensors and add the label"""
    for i, frame in enumerate(tqdm(load_single_images(video_path), "Save Frames as Tensors")):
        frame = cut_bottom(frame, picture_bottom_offset)
        frame = sample_down(frame, picture_final_size)

        if save_as_png:
            cv2.imwrite(save_png_path + "{:05d}.png".format(i), frame)

        frame = transforms.ToTensor()(frame)  # .unsqueeze(0)
        # print(frame.shape)
        torch.save(frame, save_path + "{:05d}.pt".format(i))


def save_flow_as_tensors(save_path, video_path, save_as_png=False, save_png_path=path_image_opt_fl):
    # load images, transform to tensors and add the label
    for i, (prev_frame, curr_frame) in enumerate(tqdm(load_double_images(video_path), "Save Opt. Flow as Tensors")):
        curr_frame = sample_down(cut_bottom(curr_frame, picture_bottom_offset), picture_opt_fl_size)
        prev_frame = sample_down(cut_bottom(prev_frame, picture_bottom_offset), picture_opt_fl_size)

        rgb_flow = calculate_opt_flow(curr_frame, prev_frame)
        # print(rgb_flow)
        if save_as_png:
            cv2.imwrite(save_png_path + "{:05d}.png".format(i + 1), rgb_flow)
        # transform image to a tensor and concat them
        rgb_flow_tensor = transforms.ToTensor()(rgb_flow)  # .unsqueeze(0)
        # print(rgb_flow_tensor.shape)
        torch.save(rgb_flow_tensor, save_path + "{:05d}.pt".format(i + 1))


# #############################################################
# CALCULATE OPTICAL FLOW
# #############################################################

def calculate_opt_flow(curr_frame, prev_frame):
    """
    Calculates the optical Flow of two images
    Source: https://www.geeksforgeeks.org/opencv-the-gunnar-farneback-optical-flow/
    """
    # Create mask
    hsv_mask = np.zeros_like(prev_frame)
    # Make image saturation to a maximum value
    hsv_mask[:, :, 1] = 255

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Optical flow is now calculated
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, pyr_scale=0.5, levels=3, winsize=6,
                                        iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
    # Compute magnitude and angle of 2D vector
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Set image hue value according to the angle of optical flow
    hsv_mask[:, :, 0] = ang * (90 / np.pi)
    # Set value as per the normalized magnitude of optical flow
    hsv_mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to rgb
    rgb_image = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2RGB)
    rgb_image = sample_down(rgb_image, picture_final_size)
    return rgb_image


# #############################################################
# TRAIN EVALUATION DICTIONARIES
# #############################################################

def generate_train_eval_dict(data_size, test_split_ratio, block_size=100, offset=None):
    if offset is None:
        offset = np.random.random_integers(0, 50)

    test_block = block_size * test_split_ratio

    all_indices = np.linspace(1, data_size, data_size, dtype=int)
    test_index = (all_indices - offset) % block_size < test_block
    train_indices = [*all_indices[test_index]]
    test_indices = [*all_indices[~test_index]]

    partition = {'train': train_indices, 'validation': test_indices}
    return partition


def generate_label_dict(label_path, data_size):
    """generate a dictionary with all indexes and their speeds"""
    labels_np_array = np.loadtxt(label_path)
    labels = {}
    for index in range(0, data_size + 1):
        labels[index] = labels_np_array[index]

    return labels


if __name__ == "__main__":
    #    i1 = cv2.imread("./data/frames/frame1.png")
    #    i2 = cv2.imread("./data/frames/frame2.png")
    #    i2_cut_down = sample_down_half(i2[:-60,:,:])
    #    flow_field = calc_of(i1, i2)
    #    cv2.imwrite("../report/imgs/frame2_original.png",i2)
    #    cv2.imwrite("../report/imgs/frame2_cut_sampled.png",i2_cut_down)
    #    cv2.imwrite("../report/imgs/frame2_flow_field.png",flow_field)
    # save_flow_as_tensors(path_tensor_opt_fl, path_raw_video)
    # save_frames_as_tensors(path_tensor_frames, path_raw_video)
    save_both(path_tensor_frames, path_tensor_opt_fl, path_raw_video)
