import cv2
import numpy as np
from matplotlib import pyplot as plt
#v = np.genfromtxt("data/raw/vel_data.txt")
#veloc = np.array([])

#for i in range(len(v)):
#    if i > 0:
#        veloc = np.append(veloc, np.linspace(v[i-1], v[i], 24, endpoint=(i==len(v)-1)))

#veloc = np.append(np.array([v[0]]), veloc)

#print(len(veloc))

a = np.genfromtxt("data/raw/new_train_label.txt")

plt.plot(a)
plt.show()

#video = cv2.VideoCapture("data/raw/VID_20210122_081356.mp4")
#length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#print(length)