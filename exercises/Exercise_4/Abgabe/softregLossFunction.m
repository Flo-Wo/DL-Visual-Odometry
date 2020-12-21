function [L,grad] = softregLossFunction(param,data,lambda)
% MKSVMLOSSFUNCTION Implements the softmax regression loss function
% Input:
%   param.W - weights 
%   param.b - bias 
%   data.X - input data
%   data.y - label of input data
%   lambda - regularzation parameter
%
% Output:
%   L - computed loss
%   grad.W - gradient w.r.t W
%   grad.b - gradient w.r.t b

[m,~] = size(data.X);
L = 0;
grad.W = zeros(size(param.W));
grad.b = zeros(size(param.b));

% ====================== YOUR CODE HERE ===================================
X = data.X;
y = data.y;
W = param.W;
b = param.b;
% get sizes of the data
[~,K] = size(W);
[m, ~] = size(X);
%% compute L
result = exp(X * W + param.b);
W_squared = W.^2;
L = -1/m * (sum(log(result(data.y) ./ sum(result, 2)))) + lambda/2 * sum(W_squared, "all");
%L = temp;
sizeB = length(b);
%% compute \partial_w L
grad.W = (-  data.X' * (data.y == linspace(1, sizeB, sizeB) - result ./ sum(result, 2)))./m + lambda.* W;
%% compute \partial_b L
grad.b = 1./m .* sum((result./sum(result, 2)),1) - 1./m .* sum(y == linspace(1,sizeB,sizeB), 1);
% =========================================================================


end