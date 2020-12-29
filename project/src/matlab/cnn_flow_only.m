addpath("data/optical_flow/test");
addpath("data/optical_flow/train");


% load params
params = load("/Users/florianwolf1/Documents/Studium/Bachelor/Semester5/ML-Matlab/project/src/params_2020_12_28__15_01_14.mat");

% build network
layers = [
    imageInputLayer([480 640 3],"Name","imageinput")
    convolution2dLayer([5 5],24,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")
%     convolution2dLayer([5 5],36,"Name","conv_2","Padding","same")
%     reluLayer("Name","relu_2")
%     convolution2dLayer([5 5],48,"Name","conv_3","Padding","same")
%     reluLayer("Name","relu_3")
%     convolution2dLayer([3 3],64,"Name","conv_4")
%     reluLayer("Name","relu_4")
%     convolution2dLayer([3 3],64,"Name","conv_5")
%     %flattenLayer("Name","flatten")
%     reluLayer("Name","relu_5")
%     fullyConnectedLayer(100,"Name","fc_1")
%     reluLayer("Name","relu_6")
%     fullyConnectedLayer(50,"Name","fc_2")
%     reluLayer("Name","relu_7")
    fullyConnectedLayer(10,"Name","fc_3")
    reluLayer("Name","relu_8")
    fullyConnectedLayer(1,"Name","fc_4")
    regressionLayer("Name","regressionoutput")];
plot(layerGraph(layers));

Xtrain = imageDatastore("data/optical_flow/train");
Xtest = imageDatastore("data/optical_flow/test");

labels = load("data/raw/train_label.txt");
Ytrain = labels(1:70);
Ytest = labels(71:99);

% Xtrain.Labels = Ytrain;
% Xtest.Labels = Ytest;

% train network, set options
miniBatchSize = 10;
validationFrequency = floor(numel(Ytrain)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{Xtest,Ytest}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(Xtrain, Ytrain,layers,options);



