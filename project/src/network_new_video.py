from network_user import process_video
from cnn.cnn_frames_convolutional import CnnFramesConv
from data_loader import DatasetFrames, DatasetOptFlo1Frames
from cnn.cnn_siamese_frames_flow import CnnSiamese
import numpy as np
from matplotlib import pyplot as plt

process_video("./data/raw/test.mp4", "./cnn/saved_models/NewSplitting/LeakyReLU_Frames_Conv.pth",
              "./cnn/saved_models/Videos/LeakyReLU_Frames_Conv", model=CnnFramesConv(), dataset_class=DatasetFrames)

process_video("./data/raw/test.mp4", "./cnn/saved_models/NewSplitting/LeakyReLU_SIAMESE.pth",
              "./cnn/saved_models/Videos/LeakyReLU_SIAMESE", model=CnnSiamese(3), dataset_class=DatasetOptFlo1Frames)

process_video("./data/raw/test.mp4", "./cnn/saved_models/NewSplitting/LeakyReLU_Frames_Conv_SitSplit.pth",
              "./cnn/saved_models/Videos/LeakyReLU_Frames_Conv_SitSplit", model=CnnFramesConv(),
              dataset_class=DatasetFrames)

process_video("./data/raw/test.mp4", "./cnn/saved_models/NewSplitting/LeakyReLU_SIAMESE_SitSplit.pth",
              "./cnn/saved_models/Videos/LeakyReLU_SIAMESE_SitSplit", model=CnnSiamese(3),
              dataset_class=DatasetOptFlo1Frames)

txt1 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_SIAMESE.txt")
txt2 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_Frames_Conv.txt")
txt3 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_SIAMESE_SitSplit.txt")
txt4 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_Frames_Conv_SitSplit.txt")

plt.plot(txt1, color="red")
plt.plot(txt2, color="blue")
plt.plot(txt3, ls="--", color="green")
plt.plot(txt4, ls="--", color="gray")
plt.show()
