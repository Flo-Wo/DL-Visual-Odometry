#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:19:53 2020

@author: florianwolf
"""


import torch
from cnn.cnn_flow_only import CNNFlowOnly

from utils_save_load import load_data

from utils_save_load import Dataset, generate_label_dict, generate_train_eval_dict

def write_txt_file(data, path):
    # from https://stackoverflow.com/questions/33686747/save-a-list-to-a-txt-file
    with open(path, "w") as output:
        for line in data:
            output.write(f"{line}\n")

def train_model(train_dataset, eval_dataset,num_input_channels, num_epochs):
    # create model
    model = CNNFlowOnly(num_input_channels)
    # create loss function and create optimizer object, we use the MSE Loss,
    # as this is used to evaluate our results in the initial challenge
    criterion = torch.nn.MSELoss()
    # starting with adam, later on maybe switching to SGD
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    # add a learning rate scheduler, to reduce the learning rate after several
    # epochs, as we did in the MNIST exercise
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,\
                factor=0.9,patience=2)
    # according to https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim
    # this reduces the lr by a factor of 0.1 if the relative decrease after 2
    # epochs is not bigger than the default threshold
    print("training starts!")
    for epoch in range(num_epochs):
        print("epoch: ",epoch+1)
        ## training part ##
        model.train()
        train_loss = 0
        eval_loss = 0
        # now iterate through training examples
        # train_dataset consists of batches of an torch data loader, including
        # the flow fields and the velocity vectors, attention the enumerator
        # also returns an integer
        for _, (flow_stack, velocity_vector) in enumerate(train_dataset):
            #flow_stack = flow_stack.squeeze(1)
            # according to https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            # we need to set the gradient to zero first
            optimizer.zero_grad()
            predicted_velocity = model(flow_stack)
            loss = criterion(predicted_velocity,velocity_vector.float())
            # print(loss)
            # print(loss.item())
            # backward propagation
            loss.backward()
            # use optimizer
            optimizer.step()
            # this actually returns the loss value
            train_loss += loss.item()
        ## evaluation part ##
        model.eval()
        for _, (flow_stack, velocity_vector) in enumerate(eval_dataset):
            #flow_stack = flow_stack.squeeze(1)
            # do not use backpropagation here, as this is the validation data
            with torch.no_grad():
                predicted_velocity = model(flow_stack)
                loss = criterion(predicted_velocity,velocity_vector.float())
                eval_loss += loss.item()
        # mean the error to print correctly
        print("train loss =",train_loss/len(train_dataset))
        print("eval loss =",eval_loss/len(train_dataset))
        # use the scheduler and the mean error
        scheduler.step(train_loss/len(train_dataset))
    # save the models weights and bias' to use it later
    torch.save(model.state_dict(),"./cnn/savedmodels/currentmodel.pth")
    print("model saved!")   

def evaluate_data_and_write_txt_file(eval_dataset, num_input_channels, txt_path):
    list_predicted_velocity = []
    criterion = torch.nn.MSELoss()
    # build a new netork
    model = CNNFlowOnly(num_input_channels)
    torch.laod(model.state_dict(),"./cnn/savedmodels/currentmodel.pth")
    # set model in evaluation mode
    model.eval()
    eval_loss = 0
    print("evaluation starts!")
    for _,(flow_stack, velocity_vector) in enumerate(eval_dataset):
        with torch.no_grad():
            predicted_velocity = model(flow_stack)
            loss = criterion(predicted_velocity,velocity_vector.float())
            eval_loss += loss.item()
            list_predicted_velocity.append(loss.item())
    # mean the complete error
    eval_loss_all = eval_loss/len(eval_dataset)
    print("The total evalutation loss is =",eval_loss_all)
    print("now writing the output into a txt file:")
    write_txt_file(list_predicted_velocity,txt_path)
    return(list_predicted_velocity)  
        
        
if __name__ == "__main__":
    # Parameters
    params = {'batch_size': 64,\
          'shuffle': True}
          #'num_workers': 6}
    #max_epochs = 100
    data_size = 20399
    partition = generate_train_eval_dict(data_size, 0.8)
    labels = generate_label_dict("./data/raw/train_label.txt",data_size)
    #train_tensor, eval_tensor = load_data("./data/tensorData/tensor_of_with_labels",0.8,32)
    
    # Generators
    training_set = Dataset(partition['train'], labels)
    train_tensor = torch.utils.data.DataLoader(training_set, **params)
    
    validation_set = Dataset(partition['validation'], labels)
    eval_tensor = torch.utils.data.DataLoader(validation_set, **params) 
    
    train_model(train_tensor, eval_tensor, 3, 25)
    