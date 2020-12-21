function [L, grads] = nnLossFunction(params,X,y,lambda)
% NNLOSSFUNCTION Implements the neural network loss function for a N-layer
% neural network which performs classification
%
% Input:
%   params(i).W - weights from layer i to layer i+1
%   params(i).b - bias from layer i to layer i+1
%   X - input m-by-n data
%   y - data label
%   lambda - hyper-parameter for regularization
%
%
% Output:
%   L - computed loss
%   grads(i).W - weight gradient from layer i to layer i+1
%   grads(i).b - bias gradient from layer i to layer i+1


L = 0;

num_layer = length(params)+1;
grads = struct([]);
% we need this to speedup  matlab, because the structure 
%cache = repmat(struct("x",1), num_layer-1, 1 );
%grads = repmat(struct("x",1), num_layer-1, 1 );
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
% Part 1: Feedforward the neural network and return the loss in the
%         variable L.
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%
% Part 3: Implement regularization with the loss function and gradients.
% -------------------------------------------------------------


% Forward pass: compute the loss
% hint: use for...loop and functions affine_forward, affine_relu_forward, 
% and softmax_loss in folder layer

% first layer, use the data set for the affine_relu and save the cache for 
% the gradient calculation
A_l = X;

%% testing
% for each other layer use the previous result
% for l=1:1:num_layer-1
%     if l ~= num_layer-1
%         [A_l,cache(l)] = affine_relu_forward(A_l, params(l).W, params(l).b);
%     else
%         %last layer does not use ReLu
%         [A_l,cache(l)] = affine_forward(A_l, params(l).W, params(l).b);
%     end
% end
for l=1:1:num_layer-1
    if l ~= num_layer-1
        [A_l_temp,cache_temp] = affine_relu_forward(A_l, params(l).W, params(l).b);
    else
        %last layer does not use ReLu
        [A_l_temp,cache_temp2] = affine_forward(A_l, params(l).W, params(l).b);
        cache_temp.affine_cache = cache_temp2;
        cache_temp.relu_cache = struct([]);
    end
    A_l = A_l_temp;
    cache(l) = cache_temp;
end

%save the cache
%[res,cache(num_layer-1)] = affine_forward(A_l, params(num_layer-1).W, params(num_layer-1).b);


% to compute the loss, use the result of the last layer and comupte the
% softmax loss value, here the regularization term is still missing!!
[L,Delta] = softmax_loss(A_l, y);
% this Delta is our first initial Delta

% Backward pass: compute gradients
% hint: use for...loop and functions affine_backward, affine_relu_backward
% in folder layer

% use derivative of the softmax loss function


% grad(num_layer-1).b = Delta;
% grad(num_layer-1).W = (cache(num_layer-1).affine_cache.A)'* Delta;
% 
% Delta_in = Delta;
% 
% for l = num_layer-2:2
%     grad(l).b = db;
%     grad(l).W = dW;
%     [Delta_in,dW, db] = affine_relu_backward(Delta_in,cache(l));
% end
% 
% grad(1).b = X;
% grad(1).W = Delta_in * X + lambda*params(1).W;

Delta_out = Delta;

for l=num_layer-1:-1:1
    if l ~= num_layer-1
        [Delta_out, grads(l).W, grads(l).b] = affine_relu_backward(Delta_out, cache(l));
    else
        % in the last layer we did not use the relu, just the affine one
        [Delta_out, grads(l).W, grads(l).b] = affine_backward(Delta_out, cache(l).affine_cache);
    end
end


%[Delta, grads(1).W, grads(1).b] = affine_relu_backward(X, cache(1));

% Regularization
% hint: use for...loop

% add regulariazation term to the loss function
reg_L = 0;
for l=1:num_layer-1
    W = params(l).W;
    grads(l).W = grads(l).W + lambda * W;
    reg_L = reg_L + sum(W.^2, "all");
end
% add regularization term
L = L + lambda/2*reg_L;
% =========================================================================


end