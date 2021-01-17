from network_user import process_video
from cnn.cnn_frames_convolutional import CnnFramesConv
from data_loader import DatasetFrames, DatasetOptFlo1Frames, DatasetOptFlo
from cnn.cnn_siamese_frames_flow import CnnSiamese
import numpy as np
from matplotlib import pyplot as plt

from cnn.cnn_flow_only_with_pooling import CNNFlowOnlyWithPooling

prodVideos = False

if prodVideos:
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



process_video("./data/raw/test.mp4", "./cnn/savedmodels/OriginalSplitting/LeakyReLU15EpochsBatchNormMaxPoolingWithDropOut.pth",
                  "./cnn/saved_models/Videos/LeakyReLU15EpochsBatchNormMaxPoolingWithDropOut",
                  model=CNNFlowOnlyWithPooling(3), dataset_class=DatasetOptFlo)

txt1 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_SIAMESE.txt")
txt2 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_Frames_Conv.txt")
txt3 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_SIAMESE_SitSplit.txt")
txt4 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_Frames_Conv_SitSplit.txt")
txt5 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU15EpochsBatchNormMaxPoolingWithDropOut.txt")

#plt.plot(txt1, color="red", label="Siamese Approach (100S)")
#plt.plot(txt2, color="blue", label="Convolutional (100S)")
#plt.plot(txt3, color="green", label="Siamese Approach (SitS)")
#plt.plot(txt4, color="pink", label="Convolutional (SitS)")

kernel_size = 50
kernel = np.ones(kernel_size) / kernel_size
txt1_convolved = np.convolve(txt1, kernel, mode='same')
txt2_convolved = np.convolve(txt2, kernel, mode='same')
txt3_convolved = np.convolve(txt3, kernel, mode='same')
txt4_convolved = np.convolve(txt4, kernel, mode='same')
txt5_convolved = np.convolve(txt5, kernel, mode='same')

plt.plot(txt1_convolved, color="black", label="Siamese Smoothed (100S)")
#plt.plot(txt2_convolved, color="gray", label="Conv. Smoothed (100S)")
plt.plot(txt3_convolved, color="gray", label="Siamese Smoothed (SitS)")
#plt.plot(txt4_convolved, color="black", label="Conv. Smoothed (SitS)")
plt.plot(txt5_convolved, color="black", label="Classical (HardS)")

plt.legend()
plt.savefig("./cnn/saved_models/Videos/LeakyReLU_ALL.eps")
plt.savefig("./cnn/saved_models/Videos/LeakyReLU_ALL.png")
plt.show()
