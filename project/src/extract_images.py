#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:42:30 2020

@author: florianwolf
"""

import cv2
import os
video = cv2.VideoCapture("data/raw/train.mp4")
success,image = video.read()
count = 1
os.chdir("data/frames")
while success and count <=100:
    cv2.imwrite("%d.png" % count, image)     # save frame as png file      
    success,image = video.read()
    if count % 100 == 0:
        print('Number: ', count)
    count += 1