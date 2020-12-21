function [p] = nnPredict(params,X)
%% NNPREDICT Predict the label of an input given a trained neural network

p = zeros(size(X,1),1);
% ====================== YOUR CODE HERE ===================================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels. i.e, 10
%
% hint: copy and paste the forward pass code in loss function, and then
% apply max function to get prediction

num_layer = length(params)+1;

[A_l,cache(1)] = affine_relu_forward(X, params(1).W, params(1).b);

% for each other layer use the previous result
for l=2:1:num_layer-1
    [A_l,cache(l)] = affine_relu_forward(A_l, params(l).W, params(l).b);
end
[val,p] = max(A_l,[],2);

% =========================================================================


end