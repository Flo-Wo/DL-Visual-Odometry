%% function to laod the data of the provided data set
addpath("./data/raw")
% load the image
v = VideoReader("train.mp4");
%frames = read(v);
labels = load("train_label.txt");

% now we use a 70/30 rule for splitting the data and save the parts in as
% mat files

num_frames = v.NumFrames;
% use random permutation of the indices
indices = 1:num_frames;
rate = 0.7 * num_frames;
rate = int64(rate);
rate_half = int62(rate/2);
disp("first train");
train_frames_first = read(v,[1, rate/2]);
train_labels_first = labels(1:rate/2);
disp("second train");
train_frames_second = read(v,[rate/2,rate]);
train_labels_second = labels(rate/2:rate);
disp("test");
test_frames = read(v, [rate+1, Inf]);
test_labels = labels(rate+1:end);



