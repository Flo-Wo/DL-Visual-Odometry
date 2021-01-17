#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15.01.2021
@author: Franz Herbst
"""

import logging, coloredlogs

import cv2
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from network_trainer import standard_model, standard_dataset_class

from data_loader import load_double_images, sample_down, cut_bottom, picture_bottom_offset, \
    picture_opt_fl_size, picture_final_size, calculate_opt_flow

coloredlogs.install()
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

# #############################################################
# Video Analyser
# #############################################################


def put_velocity_error_on_frame(frame, prediction, **kwargs):
    # set some important constants
    font = cv2.FONT_HERSHEY_SIMPLEX
    velo_color = (25, 255, 25)
    pred_color = (255, 25, 25)
    err_color = (25, 25, 255)
    fontScale = 1.1
    thickness = 2
    upper_offset = 40
    line_offset = 30
    right_offset = 8

    # check, whether the real velocity is given, or not, via varargs
    if "velocity" in kwargs:
        velocity = kwargs["velocity"]
        error = np.abs(velocity - prediction)
        velo_position = (right_offset, upper_offset)
        pred_position = (right_offset, upper_offset + line_offset)
        err_position = (right_offset, upper_offset + 2 * line_offset)
        pred = "pred (m/s): " + "{:2.3f}".format(prediction)
        velo = "speed (m/s): " + "{:2.3f}".format(velocity)
        err = "error (m/s): " + "{:2.3f}".format(error)

        frame_labeled = cv2.putText(frame, velo, velo_position, font,
                                    fontScale, velo_color, thickness)

        frame_labeled = cv2.putText(frame_labeled, err, err_position,
                                    font, fontScale, err_color, thickness)

        frame_labeled = cv2.putText(frame_labeled, pred, pred_position,
                                    font, fontScale, pred_color, thickness)
    else:
        pred_position = (right_offset, upper_offset)
        pred = "pred (m/s): " + "{:2.3f}".format(prediction)
        frame_labeled = cv2.putText(frame, pred, pred_position, font,
                                    fontScale, pred_color, thickness)

    return frame_labeled


def plot_velocity_chart(data_file, label="Data", color="gray", kernel_size=None, kernel_color="black"):
    velocities = np.genfromtxt(data_file)
    plt.plot(velocities, color=color, ls=None, label=label)

    if kernel_size is not None:
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved = np.convolve(velocities, kernel, mode='same')
        plt.plot(data_convolved, color=kernel_color, ls="-",
                 label="Smoothed")


def process_video(path_video, model_file, save_to, produce_video=False, label_path=None,
                  model=standard_model, dataset_class=standard_dataset_class):
    if label_path is not None:
        label = True
        theory_velocity = np.loadtxt(label_path)
    else:
        label = False

    # load like
    # https://stackoverflow.com/questions/49941426/attributeerror-collections-ordereddict-object-has-no-attribute-eval
    model.load_state_dict(torch.load(model_file))
    model.eval()

    velocities = np.array([])

    if produce_video:
        video_label = cv2.VideoWriter(save_to + ".mp4", 0x7634706d, 20, (640, 480))

    for count, (prev_frame, org_frame) in enumerate(tqdm(load_double_images(path_video), "Process Video")):
        curr_frame = sample_down(cut_bottom(org_frame, picture_bottom_offset), picture_opt_fl_size)
        prev_frame = sample_down(cut_bottom(prev_frame, picture_bottom_offset), picture_opt_fl_size)

        # FLOW
        rgb_flow = calculate_opt_flow(curr_frame, prev_frame)
        # transform image to a tensor and concat them
        rgb_flow_tensor = transforms.ToTensor()(rgb_flow)
        rgb_flow_tensor = rgb_flow_tensor.unsqueeze(0)

        # FRAME
        curr_frame = sample_down(curr_frame, picture_final_size)
        curr_frame = transforms.ToTensor()(curr_frame)
        curr_frame = curr_frame.unsqueeze(0)
        prev_frame = sample_down(prev_frame, picture_final_size)
        prev_frame = transforms.ToTensor()(prev_frame)
        prev_frame = prev_frame.unsqueeze(0)

        # logging.debug(frame.size())
        # logging.debug(rgb_flow_tensor.size())

        with torch.no_grad():
            # predicted_velocity = model(rgb_flow_tensor, frame)

            predicted_velocity = model(*dataset_class.get_images(prev_frame, curr_frame, rgb_flow_tensor))
            velocities = np.append(velocities, predicted_velocity)

        if produce_video:
            if label:
                frame_labeled = put_velocity_error_on_frame(org_frame, predicted_velocity,
                                                            velocity=(theory_velocity[count] + theory_velocity[
                                                                count + 1]) / 2)
            else:
                frame_labeled = put_velocity_error_on_frame(org_frame, predicted_velocity)

            video_label.write(frame_labeled)

    if produce_video:
        video_label.release()

    cv2.destroyAllWindows()

    #plt.plot(velocities)
    #plt.show()
    np.savetxt(save_to + ".txt", velocities)
