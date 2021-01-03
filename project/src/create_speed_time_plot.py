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

plt.plot(time, velocity_vector,"b-")
plt.xlabel("time [s]")
plt.ylabel("velocity [m/s]")
plt.show()