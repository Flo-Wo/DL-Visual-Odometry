function [L, grad] = lossFunctionLogistic(param, X, y)
%LOSSFUNCTIONLOGISTIC Compute loss and gradient for logistic regression
%   L = LOSSFUNCTIONLOGISTIC(param, X, y) computes the loss of using param 
%   as the parameter for logistic regression and the gradient of the loss
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
L = 0;
grad = zeros(size(param));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the loss of a particular choice of param.
%               You should set L to the loss.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the loss w.r.t. each parameter in param
%
% Note: grad should have the same dimensions as param
%
hX = sigmoid(X * param);

L = (-1) * 1/m * sum(y.*log(hX) + (1-y).* log(1 - hX)); 
grad = 1/m * sum((hX - y) .* X)';
% =============================================================

end
