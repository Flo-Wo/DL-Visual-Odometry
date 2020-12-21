function [param] = gradesc(param,grad,lr)
% Implement gradient descent algorithm
% Input:
%   param.W - weights 
%   param.b - bias 
%   lr - learning rate
%   grad.W - gradient w.r.t W
%   grad.b - gradient w.r.t b
%
% Output:
%   param.W - updated weights 
%   param.b - updated bias 


% ====================== YOUR CODE HERE ======================
W = param.W;
b = param.b;
param.W = W - lr * grad.W;
param.b = b - lr * grad.b;
% ============================================================
end