%% Question 2: Logistic Regression (10pt)
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     plotData.m
%     sigmoid.m
%     lossFunctionLogistic.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('Q2data.txt');
X = data(:, [1, 2]); y = data(:, 3);

%% ==================== Part 1: Plotting ====================
%  We start the exercise by first plotting the data to understand the 
%  the problem we are working with.

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotData(X, y);

% Put some labels
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;
%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ============ Part 2: Compute loss and Gradient ============
%  In this part of the exercise, you will implement the loss and gradient
%  for logistic regression. You neeed to complete the code in 
%  lossFunction.m

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_param = zeros(n + 1, 1);

% Compute and display initial loss and gradient
[loss, grad] = lossFunctionLogistic(initial_param, X, y);

fprintf('Loss at initial param (zeros): %f\n', loss);
fprintf('Gradient at initial param (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ============= Part 3: Optimizing using fminunc  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters param.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 600);

%  Run fminunc to obtain the optimal param
%  This function will return param and the loss 
[param, loss] = ...
	fminunc(@(t)(lossFunctionLogistic(t, X, y)), initial_param, options);

% Print param to screen
fprintf('Loss at param found by fminunc: %f\n', loss);
fprintf('param: \n');
fprintf(' %f \n', param);

% Plot Boundary
plotDecisionBoundary(param, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted', 'prediction fit')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
%  Furthermore, you will compute the training and test set accuracies of 
%  our model.
%
%  Your task is to complete the code in predict.m

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 

prob = sigmoid([1 45 85] * param);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob);

% Compute accuracy on our training set
p = predict(param, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

