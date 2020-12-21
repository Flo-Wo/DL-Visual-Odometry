function accuracy=multi_classifier_accuracy(param, X,y)
m = length(y);

[~,labels] = max(X*param.W+param.b, [], 2);

correct=sum(y == labels);
accuracy = correct / m;
end