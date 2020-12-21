function [params] = randInitializeParams(layer_unit)
% randInitializeParams randomly initialize the weights and zero 
% initialize biases, all the parameters save in structure params.
%

rng(1); 

params = ([]);
epsilon = 1e-3;

for i = 1:size(layer_unit,2)-1
    params(i).W = epsilon * rand(layer_unit{i},layer_unit{i+1});
    params(i).b = zeros(1,layer_unit{i+1});
end


end