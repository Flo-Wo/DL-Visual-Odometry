#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15.01.2021
@author: Franz Herbst
"""

import coloredlogs
import logging
import pickle

import numpy as np
import numpy
import torch
from tqdm import tqdm

from os import path

from cnn.cnn_siamese_frames_flow import CnnSiamese
from data_loader import DatasetOptFlo1Frames
# time module, to get date and time

from matplotlib import pyplot as plt
<<<<<<< HEAD
from data_loader import generate_label_dict, generate_train_eval_dict,\
    load_double_images, sample_down, cut_bottom, picture_bottom_offset,\
    picture_opt_fl_size, picture_final_size, \
    calculate_opt_flow#, DatasetOptFlo


# from cnn.cnn_flow_only_with_pooling import CNNFlowOnlyWithPooling
# from cnn.cnn_flow_only import CNNFlowOnly
=======

import time

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
>>>>>>> net_trainer_approach

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

<<<<<<< HEAD
class NetworkTrainer:
    data_size = 20399

    # # model_class = DatasetOptFlo2Frames
    # # network_class = CnnSiamese

    # loader_class = DatasetOptFlo
    # network_class = CNNFlowOnlyWithPooling

    def __init__(self, data_size, loader_class, network_class):
        self.data_size = data_size
        self.loader_class = loader_class
        self.network_class = network_class

    # #############################################################
    # Configure Data Loaders
    # #############################################################

    def configure_data_loader(self, labels_path, tsr, bs, dl_params,
                              new_splitting=True):
        labels = generate_label_dict(labels_path, self.data_size)
        partitions = generate_train_eval_dict(self.data_size, tsr,
                                              block_size=bs, offset=0,
                                              new_split=new_splitting)

        training_set = self.loader_class(partitions['train'], labels)
        validation_set = self.loader_class(partitions['validation'], labels)

        train_tensor = torch.utils.data.DataLoader(training_set, **dl_params)
        eval_tensor = torch.utils.data.DataLoader(validation_set, **dl_params)

        return train_tensor, eval_tensor

    def train_model(self, train_dataset, eval_dataset, num_input_channels,
                    num_epochs, save_file):
        # create model
        model = self.network_class(num_input_channels)
        # create loss function and create optimizer object, we use the MSE Loss,
        # as this is used to evaluate our results in the initial challenge
        criterion = torch.nn.MSELoss()
        # starting with adam, later on maybe switching to SGD

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        #optimizer = torch.optim.ASGD(model.parameters(), lr=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                factor=0.9,
                                                                patience=1)

        # lr_lambda = lambda epoch: 0.6
        # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1, verbose=True)
       
        # create logger
        logger = logging.getLogger("train_logger")
        logger.addHandler(logging.FileHandler(f'./cnn/train_logs/{save_file}.log', mode='w'))

        metadata = np.zeros([num_epochs, 4])

        for epoch in range(num_epochs):
            logging.info("Epoch: " + str(epoch + 1))
            ## training part ##
            model.train()
            train_loss = 0
            eval_loss = 0

            # for _, (flow_stack, normal_stack, velocity_vector) in enumerate(tqdm(train_dataset, "Train")):
            for _, (*a, velocity_vector) in enumerate(tqdm(train_dataset, "Train")):
                # flow_stack = flow_stack.squeeze(1)
                # according to https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                # we need to set the gradient to zero first
                optimizer.zero_grad()
                # predicted_velocity = model(flow_stack, normal_stack)
                predicted_velocity = model(*a)
                loss = criterion(predicted_velocity, velocity_vector.float())
                # backward propagation
                loss.backward()
                # use optimizer
                optimizer.step()
                # this actually returns the loss value
                train_loss += loss.item()
            ## evaluation part ##
            # print("evaluation...")
            model.eval()
            for _, (*a, velocity_vector) in enumerate(tqdm(eval_dataset,
                                                           "Evaluate")):
                # for _, (flow_stack, normal_stack, velocity_vector) in enumerate(tqdm(eval_dataset, "Evaluate")):
                # flow_stack = flow_stack.squeeze(1)
                # do not use backpropagation here, as this is the validation data
                with torch.no_grad():
                    # predicted_velocity = model(flow_stack, normal_stack)
                    predicted_velocity = model(*a)
                    loss = criterion(predicted_velocity, velocity_vector.float())
                    eval_loss += loss.item()
            # mean the error to print correctly
            print("\nTraining Loss: " + str(train_loss / len(train_dataset)))
            print("Eval Loss: " + str(eval_loss / len(eval_dataset)))

            metadata[epoch, :] = np.array([epoch, train_loss / len(train_dataset), eval_loss / len(eval_dataset),
                                           optimizer.param_groups[0]['lr']])
            # create logger dict, to save the data into a logger file
            log_dict = {"epoch": epoch+1,
                        "train_epoch_loss": train_loss/len(train_dataset),
                        "eval_epoch_loss": eval_loss/len(eval_dataset),
                        "lr": optimizer.param_groups[0]['lr']}
            logger.info('%s', log_dict)
            # use the scheduler and the mean error
            scheduler.step()#train_loss / len(train_dataset)

        # save the models weights and bias' to use it later
        save_path = "./cnn/savedmodels/NewSplitting/"
        torch.save(model.state_dict(), save_path + save_file + ".pth")
        np.savetxt(save_path + save_file + "_METADATA.txt", metadata, delimiter=",")


        return metadata

    def process_video(self, path_video, model_file, num_input_channels,
                      save_to, produce_video=False, label_path=None):
        if label_path is not None:
            label = True
            theory_velocity = np.loadtxt(label_path)
        else:
            label = False
=======
# parameters for data loader
standard_loader_params = {'batch_size': 64 , 'shuffle': True}
>>>>>>> net_trainer_approach

standard_dataset_class = DatasetOptFlo1Frames


def setup_data_loader(dataset_class, splitting, labels, params=standard_loader_params, test_ids=None, test_labels=None):
    training_set = dataset_class(splitting['train'], labels)
    train_tensor = torch.utils.data.DataLoader(training_set, **params)

    validation_set = dataset_class(splitting['validation'], labels)
    validation_tensor = torch.utils.data.DataLoader(validation_set, **params)

    if test_labels is not None and test_ids is not None:
        testtr_set = dataset_class([*splitting['train'], *test_ids["train"]], labels, test_labels)
        train_tensor = torch.utils.data.DataLoader(testtr_set, **params)

        test_set = dataset_class(test_ids["test"], labels, test_labels)
        test_tensor = torch.utils.data.DataLoader(test_set, **params)
    else:
        test_tensor = None

    return train_tensor, validation_tensor, test_tensor


<<<<<<< HEAD
# if __name__ == "__main__":
#     path_labels = "./data/raw/train_label.txt"

#     network_save_file = "leakyReLU8EpochsBatchNormMaxPooling"

#     test_split_ratio = 0.8
#     block_size = 3400

#     dataLoader_params = {'batch_size': 64, 'shuffle': True}

#     nwt = NetworkTrainer(20399, DatasetOptFlo, CNNFlowOnlyWithPooling)


#     tr_tensor, eval_tensor = nwt.configure_data_loader(path_labels,
#                             test_split_ratio, block_size, dataLoader_params)
#     metadata = nwt.train_model(tr_tensor, eval_tensor, 3, 12, network_save_file)

#     #nwt.plot_velocity_chart("data/raw/train_predicts.txt", label="Data Siamese", color="red")
#     #nwt.plot_velocity_chart("data/raw/train_predicts_2.txt", label="Leaky Relu", color="orange", kernel_size=100)
#     #nwt.plot_velocity_chart("data/raw/train_label.txt", label="Leaky Relu", color="green")

#     #plt.legend()
#     #plt.show()

#     #nwt.process_video("data/raw/train.mp4", "./cnn/savedmodels/LeakyReLU15EpochsBatchNormMaxPoolingWithDropOut.pth", 3,
#     #                  "data/raw/train_predicts_2")
=======
# #############################################################
# Network Trainer
# #############################################################

# standard model
standard_model = CnnSiamese(3)

# create loss function and create optimizer object, we use the MSE Loss,
# as this is used to evaluate our results in the initial challenge
standard_criterion = torch.nn.MSELoss()

# standard optimizer
standard_optimizer = torch.optim.Adam(standard_model.parameters(), lr=1e-4)

# add a learning rate scheduler, to reduce the learning rate after several
# epochs, as we did in the MNIST exercise
standard_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(standard_optimizer, factor=0.9, patience=1)



# path to save trained models to
network_folder = "./cnn/saved_models/"
# path to save log files to
logging_folder = "./cnn/train_logs/"


def save_splitting(splitting, name):
    with open(network_folder + name + ".splitting.pkl", 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(splitting, f)


def load_splitting(name):
    with open(network_folder + name+ ".splitting.pkl", 'rb') as f:
        splitting = pickle.load(f)
    return splitting


def network_exists(name):
    return path.exists(network_folder + name + ".pth")


def train_network(train_tensor, validation_tensor, num_epochs, save_file, model=standard_model,
                  criterion=standard_criterion, optimizer=standard_optimizer, scheduler=standard_scheduler,
                  overwrite=False, test_tensor=None):
    """Trains the network"""

    # create logger
    logger = logging.getLogger("train_logger")
    logger.addHandler(logging.FileHandler(logging_folder + f'{save_file}-' + '.log', mode='w'))

    if path.exists(network_folder + save_file + ".pth") and overwrite is False:
        model.load_state_dict(torch.load(network_folder + save_file + ".pth"))
        all_results = np.load(network_folder + save_file + ".epochs.npy")
        prev_epochs = all_results.shape[2]

        with open(network_folder + save_file + ".scheduler.pkl", 'rb') as f:  # Python 3: open(..., 'wb')
            scheduler = pickle.load(f)
        with open(network_folder + save_file + ".optimizer.pkl", 'rb') as f:  # Python 3: open(..., 'wb')
            optimizer = pickle.load(f)
    else:
        all_results = None
        prev_epochs = 0

    for epoch in range(num_epochs):
        # training part
        model.train()
        train_loss = 0
        train_loss_list = []

        train_results = np.empty([0, 4])

        for _, (*image_stacks, velocity_vector, ids) in \
                enumerate(tqdm(train_tensor, "Epoch {:02d}: Train".format(prev_epochs + epoch + 1))):
            # according to https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            # we need to set the gradient to zero first
            optimizer.zero_grad()
            predicted_velocity = model(*image_stacks)

            r = np.append(ids.numpy()[:, np.newaxis], predicted_velocity.detach().numpy()[:, np.newaxis], axis=1)
            r = np.append(r, velocity_vector.numpy()[:, np.newaxis], axis=1)
            r = np.append(r, np.ones(len(velocity_vector))[:, np.newaxis], axis=1)
            train_results = np.append(train_results, r, axis=0)

            #print(train_results)

            loss = criterion(predicted_velocity, velocity_vector.float())
            # backward propagation
            loss.backward()
            # use optimizer
            optimizer.step()
            # this actually returns the loss value
            train_loss += loss.item()
            train_loss_list.append(loss.item())

        train_loss = np.sqrt(train_loss / len(train_tensor))
        logging.info("Training Loss: {:12.3f}".format(train_loss))

        # evaluation part
        model.eval()
        validation_loss = 0

        for _, (*image_stacks, velocity_vector, ids) in \
                enumerate(tqdm(validation_tensor, "Epoch {:02d}: Evaluate".format(prev_epochs + epoch + 1))):
            # do not use backpropagation here, as this is the validation data
            with torch.no_grad():
                # predicted_velocity = model(flow_stack, normal_stack)
                predicted_velocity = model(*image_stacks)

                r = np.append(ids.numpy()[:, np.newaxis], predicted_velocity.detach().numpy()[:, np.newaxis], axis=1)
                r = np.append(r, velocity_vector.numpy()[:, np.newaxis], axis=1)
                r = np.append(r, np.zeros(len(velocity_vector))[:, np.newaxis], axis=1)
                train_results = np.append(train_results, r, axis=0)

                loss = criterion(predicted_velocity, velocity_vector.float())
                validation_loss += loss.item()
        # mean the error to print correctly

        validation_loss = np.sqrt(validation_loss / len(validation_tensor))
        logging.info("Validation Loss: {:12.3f}".format(validation_loss))

        test_loss = -1
        if test_tensor is not None:
            for _, (*image_stacks, velocity_vector, ids) in \
                    enumerate(tqdm(test_tensor, "Epoch {:02d}: Test".format(prev_epochs + epoch + 1))):
                # do not use backpropagation here, as this is the validation data
                with torch.no_grad():
                    # predicted_velocity = model(flow_stack, normal_stack)
                    predicted_velocity = model(*image_stacks)

                    r = np.append(ids.numpy()[:, np.newaxis], predicted_velocity.detach().numpy()[:, np.newaxis], axis=1)
                    r = np.append(r, velocity_vector.numpy()[:, np.newaxis], axis=1)
                    r = np.append(r, 2*np.ones(len(velocity_vector))[:, np.newaxis], axis=1)
                    train_results = np.append(train_results, r, axis=0)

                    loss = criterion(predicted_velocity, velocity_vector.float())
                    test_loss += loss.item()
            # mean the error to print correctly

            test_loss = np.sqrt(test_loss / len(test_tensor))
            logging.info("Test Loss: {:12.3f}".format(test_loss))

        train_results = train_results[train_results[:, 0].argsort()]

        if all_results is None:
            all_results = train_results[:, :, np.newaxis]
        else:
            all_results = np.append(all_results, train_results[:, :, np.newaxis], axis=2)

        np.save(network_folder + save_file + ".epochs.npy", all_results)
        # create logger dict, to save the data into a logger file
        log_dict = {"epoch": prev_epochs + epoch + 1,
                    "train_epoch_loss (variance)": train_loss,
                    "eval_epoch_loss (variance)": validation_loss,
                    "test_epoch_loss (variance)": test_loss,
                    "lr": get_lr(optimizer)}
        logger.info('%s', log_dict)
        # use the scheduler and the mean error
        scheduler.step(train_loss**2)

        with open(network_folder + save_file + ".scheduler.pkl", 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(scheduler, f)
        with open(network_folder + save_file + ".optimizer.pkl", 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(optimizer, f)

        # save the models weights and bias' to use it later
        torch.save(model.state_dict(), network_folder + save_file + ".pth")
        logging.info("Model saved!")
    
    return network_folder + save_file, logging_folder + f'{save_file}.log'
>>>>>>> net_trainer_approach
