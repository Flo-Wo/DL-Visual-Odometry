function [model] = svm_quadprog(K, y, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% training svm by quadratic programming
% Input:
%   K - kernel matrix
%   params - parameters
% Output:
%   model - svm model
%  edit by H.Lin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eps = 1e-3;
[Nf,~] = size(K);

K = K.*(y*y');
K = (K+K')/2.;

% Quadratic programming
opts = optimset('Algorithm','interior-point-convex','Display','off');
alpha = quadprog(K, -ones(1,Nf), [], [], y', 0, zeros(Nf, 1), params.C*ones(Nf, 1),[],opts);

% Find out support vectors
inde = find(alpha >= params.C - eps); % In the margin
inds = intersect(find(alpha > eps),find(alpha < params.C - eps)); % on the boundary of margin
indo = find(alpha <= eps);

g = K*alpha;
b = sum(g(inds))/length(inds);

model.alpha = alpha;
model.b = b;
model.inde = inde;
model.inds = inds;
model.indo = indo;

end