%% Question 2: multi-class svm (5pt)
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the second
%  question of the exercise which covers regularized multi-class svm.
%
%  You will need to complete the following functions in this exericse:
%
%     gradesc.m
%     mksvmLossFunction.m
%     softregLossFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
%

%% Initialization
clear; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%  train.X and test.X will contain the training and testing images.
%   Each matrix has size [m,n] where:
%      m is the number of examples.
%      n is the number of pixels in each image.
%  train.y and test.y will contain the corresponding labels (0 to 9).
binary_digits = false;
[train,test] = load_mnist(binary_digits);
train.y = train.y+1; % make labels 1-based.
test.y = test.y+1; % make labels 1-based.


% Randomly select 100 data points to display
m = size(train.X,1);
rand_indices = randperm(m);
sel = train.X(rand_indices(1:100),:);

%displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
%pause;



%% ================ Part 2: SVM Loss function ============================
%  In this part of the exercise, you will implement the loss function of 
%  multi-class svm classification for the handwritten digit dataset.
%


[m,n] = size(train.X);
numClasses = 10;
Delta = 1;
lambda = 1e-2;

rng(1);
param.W = rand(n,numClasses);
param.b = rand(1,numClasses);

[L,grad] = mksvmLossFunction(param,train,Delta,lambda);
fprintf('\nYour loss is %f, the loss should be 37.408378 \nif your code is correct.\n',L);
fprintf('Program paused. Press enter to continue.\n\n');
%pause;



%% =============== Part 3: Stochastic Gradient Descent Training ==========
% In this part of the exercise, you will implement the stochastic gradient
% descent for training. If you have implmented it in the first part of
% exercise, run it directly. 
% 

% Set up hpyerparameters
hyperparam.model = 'svm';
hyperparam.Delta = 1;
hyperparam.learning_rate = 5e-2;
hyperparam.lambda = 5e-6;
hyperparam.num_iters = 1;%5000
hyperparam.batch_size = 200;

% Initialize parameters
param.W = zeros(n,numClasses);
param.b = zeros(1,numClasses);

tic
[param,L_history] = trainSGD(param,train,hyperparam);
fprintf('Optimization took %f seconds.\n', toc);
figure;
plot(L_history);
xlabel('Iteration','FontSize',20);
ylabel('Loss','FontSize',20);



%% ===================== Part 4: Prediction ==============================
%  If your loss function is correct, you will achieve around 93.1% training
%  accuracy and 92.3% test accuracy.


acc = multi_classifier_accuracy(param,train.X,train.y);
fprintf('\nTraining Set Accuracy %f\n',acc*100);

acc = multi_classifier_accuracy(param,test.X,test.y);
fprintf('\nTest Set Accuracy %f\n',acc*100);

