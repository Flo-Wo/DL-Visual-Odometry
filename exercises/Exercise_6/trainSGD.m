function [params,L_history] = trainSGD(params,X,y,learning_rate,...
    lambda,num_iters,batch_size)
%TRAINSGD implements mini-batch stochastic gradient descent (SGD)
% Input:
%   params - initial weights for neural networks
%   X - m-by-n training data
%   y - groundtruth for training data
%   learning_rate - learning rate for SGD
%   lambda - hyperparameter for regularization
%   num_iters - number of iterations
%   batch_size - batch size for SGD
% Output:
%   params - optimal weights for neural network
%   L_history - history of loss output


[m,~] = size(X);
iterations_per_epoch = max(m / batch_size, 1);
L_history = [];
num_layer = length(params)+1;

for iter = 1:num_iters
    
    p = randperm(m,batch_size);
    X_batch = X(p,:);
    y_batch = y(p,:);
    
    [L,grads] = nnLossFunction(params,X_batch,y_batch,lambda);
    L_history = cat(1,L_history,L);
    
    % ====================== YOUR CODE HERE ===============================
    % Update all W and b
    % hint: use for...loop
    
    for l=1:num_layer-1
        params(l).W = params(l).W - learning_rate .* grads(l).W;
        %norm(params(l).W-params(l).W - learning_rate * grads(l).W,"fro")
        params(l).b = params(l).b - learning_rate .* grads(l).b;
        %norm(params(l).b - params(l).b - learning_rate * grads(l).b)
    end
    % =====================================================================
    
    
    if mod(iter,100)==0
        disp(['iteration ' num2str(iter) '/' num2str(num_iters) ': loss ' num2str(L)]);
    end
    
    if mod(iter,iterations_per_epoch)==0 
        learning_rate = learning_rate * 0.95; % decay learning rate after every epoch
    end
    
end


end