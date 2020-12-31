% Add file paths with test and trainingdata
addpath("data/optical_flow/test");
addpath("data/optical_flow/train");

%load params
file = fopen("data/raw/train_label.txt");
input = textscan(file, "%f", 100);
labels = input{1};
fclose(file);

data_Y_train = labels(1:75);
data_Y_test = labels(76:100);

% Load the images
data_X_train = imageDatastore("./data/optical_flow/train", ...
                              'IncludeSubfolders', true, ...
                              'FileExtensions','.png', ...
                              'LabelSource', 'foldernames');
data_X_test = imageDatastore("./data/optical_flow/test", ...
                              'IncludeSubfolders', true, ...
                              'FileExtensions','.png', ...
                              'LabelSource', 'foldernames');

ttds = tabularTextDatastore("./data/optical_flow/train/labels.txt");
                          
train_dat = combine(data_X_train, data_Y_train);
                          
% Import the layers
layers = [
    imageInputLayer([480 640 3], "Name", "ImageInput", ...
                                 "Normalization","none")
    convolution2dLayer([3 3], 32, "Name", "conv", "Padding", "same")
    reluLayer("Name", "relu")
    fullyConnectedLayer(1, "Name", "fc")
    regressionLayer("Name", "regressionoutput")];

%plot(layerGraph(layers));

% set options
miniBatchSize = 10;

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'Verbose',false, ...
    'Plots','training-progress');
%validationFrequency = floor(numel(Ytrain)/miniBatchSize);
%options = trainingOptions('sgdm', ...
%    'MiniBatchSize',miniBatchSize, ...
%    'MaxEpochs',30, ...
%    'InitialLearnRate',1e-3, ...
%    'LearnRateSchedule','piecewise', ...
%    'LearnRateDropFactor',0.1, ...
%    'LearnRateDropPeriod',20, ...
%    'Shuffle','every-epoch', ...
%    'ValidationData',{Xtest,Ytest}, ...
%    'ValidationFrequency',validationFrequency, ...
%    'Plots','training-progress', ...
%    'Verbose',false);

whos data_X_train

% train network

%% this line throws an error and we do not know how to solve it
net = trainNetwork(train_dat, layers, options);