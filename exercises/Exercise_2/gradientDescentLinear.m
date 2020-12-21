function [param, L_history] = gradientDescentLinear(X, y, param, alpha, num_iters)
%GRADIENTDESCENTLINEAR Performs gradient descent to learn param
%   param = GRADIENTDESCENTMULTI(x, y, param, alpha, num_iters) updates param by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
L_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               param. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the loss function (computeLossLinear) and gradient here.
    %
    hX = X * param;
    param = param - alpha / m * sum((hX - y) .* X)';

    % ============================================================

    % Save the loss L in every iteration    
    L_history(iter) = computeLossLinear(X, y, param);

end
end
