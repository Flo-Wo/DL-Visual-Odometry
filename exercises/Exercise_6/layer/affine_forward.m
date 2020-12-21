function [Z,cache] = affine_forward(A,W,b)
% Implement feedforward function Z = WA + b
%
% Input: 
% A - Activation, for input layer, A = x
% W - Weights
% b - Bias
% 
% Output:
% Z - result of WA + b
% cache - store W, A, and b for backward use.  
%
Z = A*W + b;
cache.A = A;
cache.W = W;
cache.b = b;



end