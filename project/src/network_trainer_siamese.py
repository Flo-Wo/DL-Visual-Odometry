#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:19:53 2020

@author: florianwolf
"""

import logging, coloredlogs

import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
#from cnn.cnn_flow_only import CNNFlowOnly
from cnn.cnn_siamese_frames_flow_2try import CnnSiamese
#from data_loader import *
from matplotlib import pyplot as plt
#from utils_save_load import Dataset_of_frames, generate_label_dict, generate_train_eval_dict, \
#    generate_train_eval_dict_new_splitting

from data_loader import DatasetOptFlo2Frames, generate_label_dict, generate_train_eval_dict, \
    load_double_images, sample_down, cut_bottom, picture_bottom_offset, picture_opt_fl_size, picture_final_size, \
    calculate_opt_flow


# #############################################################
# LOGGING INITIALISATION
# #############################################################

coloredlogs.install()
logging.basicConfig(level=logging.DEBUG)

# #############################################################
# IMPORTANT CONSTANTS
# #############################################################

data_size = 20399

path_labels = "./data/raw/train_label.txt"
network_save_file = "./cnn/savedmodels/LeakyReLU_SIAMESE_2"

test_split_ratio = 0.8
block_size = 3400

dataLoader_params = {'batch_size': 64, 'shuffle': True}

model_class = DatasetOptFlo2Frames
network_class = CnnSiamese

# #############################################################
# Configure Data Loaders
# #############################################################

def configure_data_loader(labels_path, size, tsr, bs, dl_params):
    labels = generate_label_dict(labels_path, size)
    partitions = generate_train_eval_dict(size, tsr, bs, offset=0)

    training_set = model_class(partitions['train'], labels)
    validation_set = model_class(partitions['validation'], labels)

    train_tensor = torch.utils.data.DataLoader(training_set, **dl_params)
    eval_tensor = torch.utils.data.DataLoader(validation_set, **dl_params)

    return train_tensor, eval_tensor


def train_model(train_dataset, eval_dataset, num_input_channels, num_epochs, save_file):
    # create model
    model = network_class(num_input_channels)
    # create loss function and create optimizer object, we use the MSE Loss,
    # as this is used to evaluate our results in the initial challenge
    criterion = torch.nn.MSELoss()
    # starting with adam, later on maybe switching to SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # add a learning rate scheduler, to reduce the learning rate after several
    # epochs, as we did in the MNIST exercise
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=1)
    # reduce learning rate each epoch by 10%
    # lr_lambda = lambda epoch: 0.6
    # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1, verbose=True)
    # according to https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim
    # this reduces the lr by a factor of 0.1 if the relative decrease after 2
    # epochs is not bigger than the default threshold
    logging.debug("Begin Training")

    metadata = np.zeros([num_epochs, 4])

    for epoch in range(num_epochs):
        logging.info("Epoch: " + str(epoch + 1))
        ## training part ##
        model.train()
        train_loss = 0
        eval_loss = 0
        # now iterate through training examples
        # train_dataset consists of batches of an torch data loader, including
        # the flow fields and the velocity vectors, attention the enumerator
        # also returns an integer
        # print("training...")
        #for _, (flow_stack, normal_stack, velocity_vector) in enumerate(tqdm(train_dataset, "Train")):
        for _, (*a, velocity_vector) in enumerate(tqdm(train_dataset, "Train")):
            # flow_stack = flow_stack.squeeze(1)
            # according to https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            # we need to set the gradient to zero first
            optimizer.zero_grad()
            #predicted_velocity = model(flow_stack, normal_stack)
            predicted_velocity = model(*a)
            loss = criterion(predicted_velocity, velocity_vector.float())
            # print(loss)
            # print(loss.item())
            # backward propagation
            loss.backward()
            # use optimizer
            optimizer.step()
            # this actually returns the loss value
            train_loss += loss.item()
        ## evaluation part ##
        # print("evaluation...")
        model.eval()
        for _, (*a, velocity_vector) in enumerate(tqdm(eval_dataset, "Evaluate")):
        #for _, (flow_stack, normal_stack, velocity_vector) in enumerate(tqdm(eval_dataset, "Evaluate")):
            # flow_stack = flow_stack.squeeze(1)
            # do not use backpropagation here, as this is the validation data
            with torch.no_grad():
                #predicted_velocity = model(flow_stack, normal_stack)
                predicted_velocity = model(*a)
                loss = criterion(predicted_velocity, velocity_vector.float())
                eval_loss += loss.item()
        # mean the error to print correctly
        logging.info("Training Loss: " + str(train_loss / len(train_dataset)))
        logging.info("Eval Loss: " + str(eval_loss / len(eval_dataset)))

        metadata[epoch, :] = np.array([epoch, train_loss / len(train_dataset), eval_loss / len(eval_dataset),
                                       optimizer.param_groups[0]['lr']])

        # use the scheduler and the mean error
        scheduler.step(train_loss / len(train_dataset))

    # save the models weights and bias' to use it later
    torch.save(model.state_dict(), save_file + ".pth")
    logging.debug("Model saved!")
    np.savetxt(save_file + "_METADATA.txt", metadata, delimiter=",")

    return metadata


def process_video(path_video, model_file, num_input_channels, save_to):
    # build a new network
    model = network_class(num_input_channels)
    # load like
    # https://stackoverflow.com/questions/49941426/attributeerror-collections-ordereddict-object-has-no-attribute-eval
    model.load_state_dict(torch.load(model_file))
    model.eval()

    velocities = np.array([])

    for _, (prev_frame, curr_frame) in enumerate(tqdm(load_double_images(path_video), "Process Video")):
        curr_frame = sample_down(cut_bottom(curr_frame, picture_bottom_offset), picture_opt_fl_size)
        prev_frame = sample_down(cut_bottom(prev_frame, picture_bottom_offset), picture_opt_fl_size)

        # SAVE FRAME
        frame = sample_down(curr_frame, picture_final_size)
        frame = transforms.ToTensor()(frame)
        frame = frame.unsqueeze(0)
        # SAVE FLOW
        rgb_flow = calculate_opt_flow(curr_frame, prev_frame)
        # transform image to a tensor and concat them
        rgb_flow_tensor = transforms.ToTensor()(rgb_flow)  # .unsqueeze(0)
        rgb_flow_tensor = rgb_flow_tensor.unsqueeze(0)

        logging.debug(frame.size())
        logging.debug(rgb_flow_tensor.size())

        with torch.no_grad():
            predicted_velocity = model(rgb_flow_tensor, frame)
            velocities = np.append(velocities, predicted_velocity)

    plt.plot(velocities)
    plt.show()
    np.savetxt(save_to, velocities)


#velocities = np.genfromtxt("data/raw/train_predicts.txt")

#kernel_size = 100
#kernel = np.ones(kernel_size) / kernel_size
#data_convolved_10 = np.convolve(velocities, kernel, mode='same')

#data_convolved_10 = np.genfromtxt("data/raw/train_label.txt")

#plt.plot(velocities, ".")
#plt.plot(data_convolved_10, "-")
#plt.show()

#velocities = np.genfromtxt("data/raw/test_predicts.txt")
#plt.plot(velocities, ".")
#plt.show()
#process_video("data/raw/train.mp4", "./cnn/savedmodels/LeakyReLU_SIAMESE.pth", 3, "data/raw/train_predicts.txt")

train_tensor, eval_tensor = configure_data_loader(path_labels, data_size, test_split_ratio, block_size, dataLoader_params)
train_loss_list, eval_loss_list, epoch_list, lr_list = train_model(train_tensor, eval_tensor, 3, 20, network_save_file)



