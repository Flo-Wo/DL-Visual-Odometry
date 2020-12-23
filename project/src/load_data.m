%% function to laod the data of the provided data set
addpath("./data")
% load the image
v = VideoReader("train.mp4");
%frames = read(v);
labels = load("train_label.txt");

% now we use a 70/30 rule for splitting the data and save the parts in as
% mat files

num_frames = v.NumFrames;
% use random permutation of the indices
indices = randperm(num_frames);
rate = 0.7 * num_frames;
train_idx = indices(1:rate);
test_idx = indices(rate+1:end);

size_train = size(train_idx)
size_test = size(test_idx)

disp("now init structs");
train_frames = struct("frame",zeros(480,640,3,"uint8"));%struct("frames",zeros(1, int64(size_train(2))));
test_frames = struct("frame",zeros(480,640,3,"uint8"));%struct("frames",cell(1, int64(size_test(2))));
j=1;
for i=train_idx
    temp = read(v,i);
    train_frames(j) = temp;
    j = j+1;
end

%train_frames = read(v,train_idx);
save("train_frames.mat","train_frames");

for i=test_idx
    test_frames(i) = read(v,i);
end

%test_frames = read(v,test_idx);
save("test_frames.mat","test_frames");

train_labels = labels(train_idx);
save("train_labels.mat","train_labels");
test_labels = labels(test_idx);
save("test_labels.mat","test_labels");