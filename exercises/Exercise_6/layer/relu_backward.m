function [Delta_out] = relu_backward(Delta_in,cache)
% Implement backward of ReLU function, i.e., f(x) = max(0,x)
Z = cache.Z;
Delta_out = Delta_in.*(Z>0);
%Delta_out = (Z>0);
end