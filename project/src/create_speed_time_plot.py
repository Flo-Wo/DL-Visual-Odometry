#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:41:15 2021

@author: florianwolf
"""

import numpy as np
import matplotlib.pyplot as plt


# function to create a speed time plot, to look a the speed distribution

velocity_vector = np.loadtxt("./data/raw/train_label.txt")

number_of_velo = len(velocity_vector)

time = np.arange(1,number_of_velo+1)

plt.plot(time, velocity_vector,"k-")
plt.xlabel("frame number")
plt.ylabel("velocity [m/s]")
plt.vlines(x = 8400, ymin = 0, ymax = max(velocity_vector), 
           colors = 'gray')
plt.vlines(x = 15000, ymin = 0, ymax = max(velocity_vector), 
           colors = 'gray')

plt.text(8400/2, max(velocity_vector), 'highway driving', horizontalalignment='center',
      verticalalignment='center')

plt.text((15000+8400)/2, max(velocity_vector), 'stop and go', horizontalalignment='center',
      verticalalignment='center')
plt.text((15000+22399)/2, max(velocity_vector), 'city driving', horizontalalignment='center',
      verticalalignment='center')
plt.show()