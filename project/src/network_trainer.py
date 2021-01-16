#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15.01.2021
@author: Franz Herbst
"""

import coloredlogs
import logging
import torch
from tqdm import tqdm

from cnn.cnn_siamese_frames_flow import CnnSiamese
from data_loader import DatasetOptFlo1Frames

# #############################################################
# LOGGING INITIALISATION
# #############################################################

coloredlogs.install()
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

# #############################################################
# Configure Data Loaders
# #############################################################

# parameters for data loader
standard_loader_params = {'batch_size': 64, 'shuffle': True}

standard_dataset_class = DatasetOptFlo1Frames


def setup_data_loader(dataset_class, splitting, labels, params=standard_loader_params):
    training_set = dataset_class(splitting['train'], labels)
    train_tensor = torch.utils.data.DataLoader(training_set, **params)

    validation_set = dataset_class(splitting['validation'], labels)
    validation_tensor = torch.utils.data.DataLoader(validation_set, **params)

    return train_tensor, validation_tensor


# #############################################################
# Network Trainer
# #############################################################

# standard model
standard_model = CnnSiamese(3)
# all models we are using
MODEL_Siamese = CnnSiamese(3)

# create loss function and create optimizer object, we use the MSE Loss,
# as this is used to evaluate our results in the initial challenge
standard_criterion = torch.nn.MSELoss()
# all criteria we are using
CRITERION_MSELoss = torch.nn.MSELoss()

# standard optimizer
standard_optimizer = torch.optim.Adam(standard_model.parameters(), lr=1e-3)
# all optimizers we are using
OPTIMIZER_Adam_Siamese = torch.optim.Adam(MODEL_Siamese.parameters(), lr=1e-3)

# add a learning rate scheduler, to reduce the learning rate after several
# epochs, as we did in the MNIST exercise
standard_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(standard_optimizer, factor=0.9, patience=1)
# all schedulers we are using
SCHEDULER_RedLROnPlateau_Siamese = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER_Adam_Siamese,
                                                                              factor=0.9, patience=1)

# path to save trained models to
network_folder = "./cnn/saved_models/NewSplitting/"
# path to save log files to
logging_folder = "./cnn/train_logs/"


def train_network(train_tensor, validation_tensor, num_epochs, save_file, model=standard_model,
                  criterion=standard_criterion, optimizer=standard_optimizer, scheduler=standard_scheduler):
    """Trains the network"""

    # create logger
    logger = logging.getLogger("train_logger")
    logger.addHandler(logging.FileHandler(logging_folder + f'{save_file}.log', mode='w'))

    for epoch in range(num_epochs):
        # training part
        model.train()
        train_loss = 0

        for _, (*image_stacks, velocity_vector) in \
                enumerate(tqdm(train_tensor, "Epoch {:02d}: Train".format(epoch + 1))):
            # according to https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            # we need to set the gradient to zero first
            optimizer.zero_grad()
            predicted_velocity = model(*image_stacks)
            loss = criterion(predicted_velocity, velocity_vector.float())
            # backward propagation
            loss.backward()
            # use optimizer
            optimizer.step()
            # this actually returns the loss value
            train_loss += loss.item()

        train_loss = train_loss / len(train_tensor)
        logging.info("Training Loss: {:1.2e}".format(train_loss))

        # evaluation part
        model.eval()
        validation_loss = 0

        for _, (*image_stacks, velocity_vector) in \
                enumerate(tqdm(validation_tensor, "Epoch {:02d}: Evaluate".format(epoch + 1))):
            # do not use backpropagation here, as this is the validation data
            with torch.no_grad():
                # predicted_velocity = model(flow_stack, normal_stack)
                predicted_velocity = model(*image_stacks)
                loss = criterion(predicted_velocity, velocity_vector.float())
                validation_loss += loss.item()
        # mean the error to print correctly

        validation_loss = validation_loss / len(validation_tensor)
        logging.info("Training Loss: {:1.2e}".format(validation_loss))

        # create logger dict, to save the data into a logger file
        log_dict = {"epoch": epoch + 1,
                    "train_epoch_loss": train_loss,
                    "eval_epoch_loss": validation_loss,
                    "lr": optimizer.param_groups[0]['lr']}
        logger.info('%s', log_dict)
        # use the scheduler and the mean error
        scheduler.step(train_loss)

    # save the models weights and bias' to use it later
    torch.save(model.state_dict(), network_folder + save_file + ".pth")

    return network_folder + save_file + ".pth", logging_folder + f'{save_file}.log'
