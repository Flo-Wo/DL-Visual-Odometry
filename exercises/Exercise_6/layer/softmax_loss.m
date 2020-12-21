function [L,Delta] = softmax_loss(scores,y)
% Given predict scores, namely, Z^(L) and label y, compute the loss and
% derivative Delta

[m,~] = size(y);
P = exp(scores);
P = P./sum(P,2);
Y = zeros(size(P));
idx = sub2ind(size(Y),1:m,y');
Y(idx) = 1;

L = -sum(sum(log(P).*Y))/m;

Delta = P - Y;

end