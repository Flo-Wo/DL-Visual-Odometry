function [Delta_out,dW,db] = affine_backward(Delta_in,cache)
% Implement backward of function Z = WA + b
%
% Input: 
% Delta_in - Temporary variable from previous layer
% cache - stored W, A, and b
% 
% Output:
% Delta_out - Temporary variable of current layer
% dW - derivative w.r.t. weights
% db - derivative w.r.t. bias
%


A = cache.A;
W = cache.W;
[m,~] = size(A);

dW = A'*Delta_in/m;%
db = sum(Delta_in)/m;%
Delta_out = Delta_in*W';

end