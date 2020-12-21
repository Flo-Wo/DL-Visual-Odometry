function [k] = gausskernel(x,y,sigma)
% Gaussian kernel, compute exp(-||x-y||^2/sigma)

[numx,~] = size(x);
[numy,~] = size(y);
k = x*y';
k = 2*k;
k = k-sum(x.^2,2)*ones(1,numy);
k = k-ones(numx,1)*sum(y.^2,2)';
k = exp(k/sigma);
%k = (k+k')/2;
end