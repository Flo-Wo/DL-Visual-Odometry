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
P = result./sum(result, 2);
I = full(sparse(1:m, y,1));
L = -sum(sum(log(P).*I))/m + lambda/2 * sum(sum(W_squared));
%% compute \partial_w L
his = -X'*(I-P)/m + lambda * W;
grad.W = his;
%% compute \partial_b L
grad.b = mean(I - P,1);
% =========================================================================
end