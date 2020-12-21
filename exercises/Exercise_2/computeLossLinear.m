function L = computeLossLinear(X, y, param)
%COMPUTELOSSLINEAR Compute loss for linear regression with multiple variables
%   L = COMPUTELOSSLINEAR(X, y, param) computes the loss of using param as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
L = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the loss of a particular choice of param
%               You should set L to the loss.
hX = X*param;
L = 1/(2.*m) * sum((hX - y).^2);
% =========================================================================

end
