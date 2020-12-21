function [Delta_out,dW,db] = affine_relu_backward(Delta_in,cache)
% Implement backward pass of FC layer, output Delta, dW, and db
%
% Input: 
% Delta_in - Temporary variable from previous layer, used for backward
% propagation
% cache - store variables from forward, see affine_relu_forward function 
% 
% Output:
% Delta_out - Temporary variable of current layer, used for backward
% propagation
% dW - derivative w.r.t. weights
% db - derivative w.r.t. bias
%


% You will use the two variables
affine_cache = cache.affine_cache;
relu_cache = cache.relu_cache;

% ====================== YOUR CODE HERE =================================
% Hint: Use affine_backward function and relu_backward function here.
% first invert the relu function, then invert the affine function
Delta_out = relu_backward(Delta_in, relu_cache); % Delta_out = f' * Delta^{l+1}
[Delta_out, dW, db] = affine_backward(Delta_out, affine_cache);
% =======================================================================

end