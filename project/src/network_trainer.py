#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:19:53 2020

@author: florianwolf
"""

import logging#, coloredlogs

import cv2
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from data_loader import generate_label_dict, generate_train_eval_dict,\
    load_double_images, sample_down, cut_bottom, picture_bottom_offset,\
    picture_opt_fl_size, picture_final_size, \
    calculate_opt_flow#, DatasetOptFlo


# from cnn.cnn_flow_only_with_pooling import CNNFlowOnlyWithPooling
# from cnn.cnn_flow_only import CNNFlowOnly

# #############################################################
# LOGGING INITIALISATION
# #############################################################

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


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

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        #optimizer = torch.optim.ASGD(model.parameters(), lr=1e-4)

        # add a learning rate scheduler, to reduce the learning rate after several
        # epochs, as we did in the MNIST exercise
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.9,
                                                               patience=1)
        # reduce learning rate each epoch by 10%
        # lr_lambda = lambda epoch: 0.6
        # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1, verbose=True)
        # according to https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim
        # this reduces the lr by a factor of 0.1 if the relative decrease after 2
        # epochs is not bigger than the default threshold
        #logging.debug("Begin Training")
        
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
            # now iterate through training examples
            # train_dataset consists of batches of an torch data loader, including
            # the flow fields and the velocity vectors, attention the enumerator
            # also returns an integer
            # print("training...")
            # for _, (flow_stack, normal_stack, velocity_vector) in enumerate(tqdm(train_dataset, "Train")):
            for _, (*a, velocity_vector) in enumerate(tqdm(train_dataset, "Train")):
                # flow_stack = flow_stack.squeeze(1)
                # according to https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                # we need to set the gradient to zero first
                optimizer.zero_grad()
                # predicted_velocity = model(flow_stack, normal_stack)
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
            scheduler.step(train_loss / len(train_dataset))

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

        # build a new network
        model = self.network_class(num_input_channels)
        # load like
        # https://stackoverflow.com/questions/49941426/attributeerror-collections-ordereddict-object-has-no-attribute-eval
        model.load_state_dict(torch.load(model_file))
        model.eval()

        vels = np.array([])

        if produce_video:
            video_label = cv2.VideoWriter(save_to + ".mp4", 0x7634706d,
                                          20, (640, 480))

        for count, (prev_frame, org_frame) in enumerate(tqdm(load_double_images(path_video), "Process Video")):
            curr_frame = sample_down(cut_bottom(org_frame, 
                                                picture_bottom_offset),
                                     picture_opt_fl_size)
            prev_frame = sample_down(cut_bottom(prev_frame,
                                                picture_bottom_offset),
                                     picture_opt_fl_size)

            # FLOW
            rgb_flow = calculate_opt_flow(curr_frame, prev_frame)
            # transform image to a tensor and concat them
            rgb_flow_tensor = transforms.ToTensor()(rgb_flow)  # .unsqueeze(0)
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
                predicted_velocity = model(self.model_class.get_images(prev_frame, curr_frame, rgb_flow_tensor))
                vels = np.append(vels, predicted_velocity)

            if produce_video:
                if label:
                    frame_labeled = self.put_velocity_error_on_frame(org_frame, predicted_velocity,
                                                        velocity=(theory_velocity[count] + theory_velocity[count+1])/2)
                else:
                    frame_labeled = self.put_velocity_error_on_frame(org_frame, predicted_velocity)
                video_label.write(frame_labeled)

        video_label.release()
        cv2.destroyAllWindows()

        plt.plot(vels)
        plt.show()
        np.savetxt(save_to + ".txt", vels)

    @classmethod
    def put_velocity_error_on_frame(cls, frame, prediction, **kwargs):
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

    @classmethod
    def plot_velocity_chart(cls, data_file, label="Data", color="gray",
                            kernel_size=None, kernel_color="black"):
        velocities = np.genfromtxt(data_file)
        plt.plot(velocities, color=color, ls=None, label=label)

        if kernel_size is not None:
            kernel = np.ones(kernel_size) / kernel_size
            data_convolved = np.convolve(velocities, kernel, mode='same')
            plt.plot(data_convolved, color=kernel_color, ls="-",
                     label="Smoothed")


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