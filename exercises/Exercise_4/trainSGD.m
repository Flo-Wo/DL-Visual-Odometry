function [param,L_history] = trainSGD(param,data,hyperparam)
%TRAINSGD implements mini-batch stochastic gradient descent (SGD)for
%multi-class svm
% Input:
%   loss_fun - loss function that is used for loss computation
%   param - initial weights for a ML model
%   data.X - m-by-n training data
%   data.y - groundtruth for training data
%   hyperparam.learning_rate - learning rate for SGD
%   hyperparam.lambda - hyperparameter for regularization
%   hyperparam.num_iters - number of iterations for SGD
%   hyperparam.batch_size - batch size for SGD
% Output:
%   param - optimal weights for ML model
%   L_history - history of loss output
%


[m,~] = size(data.X);
iterations_per_epoch = max(m / hyperparam.batch_size, 1);
L_history = [];


for iter = 1:hyperparam.num_iters
    p = randperm(m,hyperparam.batch_size);
    data_batch.X = data.X(p,:);
    data_batch.y = data.y(p,:);
    if strcmp(hyperparam.model,'svm')
        [L,grad] = mksvmLossFunction(param,data_batch,hyperparam.Delta,hyperparam.lambda);
    elseif strcmp(hyperparam.model,'softmax')
        [L,grad] = softregLossFunction(param,data_batch,hyperparam.lambda);
    end
    
    L_history = cat(1,L_history,L);
    
    param = gradesc(param,grad,hyperparam.learning_rate);
    
    
    if mod(iter,100)==0
        disp(['iteration ' num2str(iter) '/' num2str(hyperparam.num_iters) ': loss ' num2str(L)]);
    end
    
    if mod(iter,iterations_per_epoch)==0 
        hyperparam.learning_rate = hyperparam.learning_rate * 0.95; % decay learning rate after every epoch
    end
    
end


end