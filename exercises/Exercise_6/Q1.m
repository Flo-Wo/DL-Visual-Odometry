%% Question 1: A N-layer Neural Network (15pt)
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  neural network exercise. You will need to complete the following functions 
%  in this exericse:
%
%     layer/affine_relu_forward.m
%     layer/affine_relu_backward.m
%     nnLossFunction.m
%     trainSGD.m
%     nnPredict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
%

%% Initialization
clear; close all; clc
addpath('layer'); % add folder path 'layer' for future use

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

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Feedforward function ================
% Implement the feedward function of FC layers in "affine_relu_forward"
% function. 

expect_result = [0.8937    1.8399    0.9837;
                 1.2209    2.4710    1.6061;
                 0.6091    1.2394    0.4272;
                 1.0878    2.0866    1.3031];

rng(1); 
A = rand(4,6);

rng(1); 
W = rand(6,3);

rng(1); 
b = rand(1,3);

[A_next,cache] = affine_relu_forward(A,W,b);

disp('Your output of A_next is:')
disp(A_next);

disp('If you code is correct, the output A_next should be:');
disp(expect_result);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Backward function ================
% Implement the backward function of FC layers in "affine_relu_backward"
% function. 
expect_result = [0.2824    0.6995    0.0691    0.4712    0.2883    0.3607;
                 0.4278    1.0239    0.0515    0.6288    0.3693    0.4308;
                 0.1204    0.4325    0.0854    0.3815    0.2530    0.3618;
                 0.3305    0.9389    0.1559    0.7370    0.4752    0.6475];


rng(1); 
Delta_in = rand(4,3);

[Delta_out,dW,db] = affine_relu_backward(Delta_in,cache);

disp('Your output of Delta_out is:')
disp(Delta_out);

disp('If you code is correct, the output Delta_out should be:');
disp(expect_result);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Setup the parameters you will use for this exercise 
% We create a 4-layer neural networks, the unit for two hidden layers are
% 200 and 100. You can have more hidden layers, however, note that more
% hidden layers, more computational time required. 

layer_unit = {784,200,100,10};


%% ================ Part 4: Loss function ==============================
% Module implementation of loss function (nnLossFunction.m).

fprintf('\nInitializing Neural Network Parameters ...\n');
params = randInitializeParams(layer_unit);

lambda = 0.5;
[L,grads] = nnLossFunction(params,train.X,train.y,lambda);
fprintf('\nYour loss is %f, the loss should be 2.317429 \nif your code is correct.\n',L);
fprintf('Program paused. Press enter to continue.\n\n');
pause;


%% =============== Part 5: Stochastic Gradient Descent Training ==========
% Given the loss function, now you will implement Stochastic Gradient
% Descent (SGD) (trainSGD.m) to train the neural networks. 
%

learning_rate = 0.5;%0.5
lambda = 5e-6;
num_iters = 10000; %10000
batch_size = 200;
tic
[params,L_history] = trainSGD(params,train.X,train.y,learning_rate,...
    lambda,num_iters,batch_size);
fprintf('Optimization took %f seconds.\n', toc);
figure;
plot(L_history);
xlabel('Iteration','FontSize',20);
ylabel('Loss','FontSize',20);
pause;

%% ================= Part 6: Implement Predict ===========================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "nnPredict" function to use the
%  neural network to predict the labels of the training set and test set. 
%  With default training, you will achieve around 100% training accuracy
%  and around 98.0% test accuracy.
%

p = nnPredict(params,train.X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == train.y)) * 100);
p = nnPredict(params,test.X);
fprintf('\nTest Set Accuracy: %f\n', mean(double(p == test.y)) * 100);


%%
rmpath('layer');
