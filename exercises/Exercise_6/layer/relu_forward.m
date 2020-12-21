function [A,cache] = relu_forward(Z)
% Implement feedforward of ReLU function, i.e., f(x) = max(0,x)
A = max(0,Z);
cache.Z = Z;
end