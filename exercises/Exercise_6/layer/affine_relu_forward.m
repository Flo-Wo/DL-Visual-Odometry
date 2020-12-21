function [A_next,cache] = affine_relu_forward(A,W,b)
% Implement forward pass of FC layer, namely, A_next = max(0, WA+b)
%
% Input: 
% A - Activation input of current layer, for input layer, A = x
% W - Weights
% b - Bias
% 
% Output:
% A_next - Activation output for next layer
% cache.affine_cache - store cache from affine layer
% cache.relu_cache - store cache from relu layer 
%
% ====================== YOUR CODE HERE =================================
% Hint: Use affine_forward function and relu_forward function here
[Z, cache.affine_cache] = affine_forward(A, W, b);
[A_next, cache.relu_cache] = relu_forward(Z);
% =======================================================================

end