# Visual odemetry using Deep Learning
In this project we try to estimate the velocity of a car using Deep Learning methods. We tried
using optical flow (Farneback pyramid method) and a siamese approach with two consecutive
frames as input.

We used the [comma ai speedchallenge](https://github.com/commaai/speedchallenge) to gather data and trained our models
on (different) splits of the test video. See one of our result videos below:

![results_siamese_10s_highway](https://user-images.githubusercontent.com/22920505/137796632-0b4b6908-1485-465e-a0ef-279ce3b918e5.gif)


## Abstract of the project
Calculating the velocity of a moving camera relative to its surrounding, the so-called visual odometry 
problem, is a very challenging task and heavily studied in the area of computer vision. Especially
in the field of self-driving cars, a fast and dependable velocity calculation is a high priority.
In this report we will give a machine learning approach to solve the visual odometry problem, using 
optical flow fields combined with convolutional neuronal networks, as well as siamese neuronal networks.
We use a data set with real world dashcam footage and even extend the data by gathering our own
driving videos.

## Content of this repo
This repo includes our source code, the final version of the technical report and our presentation. Under `project/presentation/demos` 
you can also find two short demo videos with results of our networks.
