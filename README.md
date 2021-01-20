# Visual odemetry using Deep Learning
In this project we try to estimate the velocity of a car using Deep Learning methods. We tried
using optical flow (Farneback pyramid method) and a siamese approach with two consecutive
frames as input.

We used the [comma ai speedchallenge](https://github.com/commaai/speedchallenge) to gather data and trained our models
on (different) splits of the test video.

## Optical flow approach
For computational reasons, we decided to cut off the last 60 pixels from the bottom, to remove a black frame inside the
car, which did not participate in the optical flow and we sampled down the frames to half the size.
We then solved the optical flow equation using the Farneback pyramid method (for parameter choices please take a look at
the report or the presentation) and sampled down the images again, to speed up the training.
As a result, we got optical flow fields like this one

IMAGE

We describe in the report, how we visualized the two dimensional flow field with as an rgb image.

