
clear
% generate two set of data
mu = [-1 -1];
sigma = [1 0; 0 1];
data1 = mvnrnd(mu,sigma,30);
y1 = ones(30,1);

mu = [1 1];
sigma = [1 0; 0 1];
data2 = mvnrnd(mu,sigma,30);
y2 = -ones(30,1);

%plot(data1(:,1),data1(:,2),'ko'); hold on;
%plot(data2(:,1),data2(:,2),'gx'); hold on;

X = cat(1,data1,data2);
y = cat(1,y1,y2);


% compute kernel matrix
params.Sigma = 5;
K = gausskernel(X,X,params.Sigma);


% solve by quadratic programming
params.C = 20;
[model] = svm_quadprog(K, y, params);

figure(1);
plot(data1(:,1),data1(:,2),'kx','MarkerSize',9); hold on;
plot(data2(:,1),data2(:,2),'k+','MarkerSize',9);
plot(X(model.inde,1),X(model.inde,2),'r*');
plot(X(model.inds,1),X(model.inds,2),'go','MarkerFaceColor','g');
hold off;

%drawfig(X, y, model,params);
