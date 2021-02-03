function [L] = lossFunctionL(h,y)
% given predicted value and ground truths, compute loss function

m = size(y,1); % number of training examples

%{
% implemented by unvectorize
L = 0;
for i = 1:m
    L = L + (h(i) - y(i)).^2;
end
L = L/(2*m);
%}


% implemented by vectorize
L = sum((h - y).^2)/(2*m);

end