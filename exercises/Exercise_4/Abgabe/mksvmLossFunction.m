function [L,grad] = mksvmLossFunction(param,data,Delta,lambda)
% MKSVMLOSSFUNCTION Implements the multi-class SVM loss function
% Input:
%   param.W - weights 
%   param.b - bias 
%   data.X - input data
%   data.y - label of input data
%   Delta - margin for multi-class
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
W = param.W;
b = param.b;
X = data.X;
y = data.y;

h = X*W + b;
h_tilde = h(y);

term = (h - h_tilde) + Delta;
term_max = max(0, term);

W_squared = W.^2;

% we sum over all and subtract the delta in the end
L = 1/m .* (sum(term_max,"all") - Delta) + lambda/2.* sum(W_squared,"all");

% Please note, I atteched to photos. When I run the progam in 6/10 cases I
% get one result and in the other 4/10 a get another one, although I did
% not change anything in the code. I don't know, how to handle this.

%% gradient calculation

%gradW = lambda .* W  - X(h+Delta > h_tilde);
%grad.W = gradW;
% =========================================================================



end