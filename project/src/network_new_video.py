from network_user import process_video
from cnn.cnn_frames_convolutional import CnnFramesConv
from data_loader import DatasetFrames, DatasetOptFlo1Frames, DatasetOptFlo, DatasetOptFlo2Frames
from cnn.cnn_siamese_frames_flow import CnnSiamese
import numpy as np
from matplotlib import pyplot as plt

from cnn.cnn_flow_only_with_pooling import CNNFlowOnlyWithPooling
from cnn.cnn_ofandframes_convolutional import CnnFramesOfConv
from cnn.cnn_supersimple import CnnSuperSimple

prodVideos = False

video = "./data/raw/test.mp4"

if prodVideos:
    process_video(video, "./cnn/saved_models/LeakyReLU_MixedSIAMESE_SitSplit.pth",
                  "./cnn/saved_models/LeakyReLU_MixedSIAMESE_SitSplit_test",
                  model=CnnSiamese(3), dataset_class=DatasetOptFlo1Frames)

    process_video(video, "./cnn/saved_models/LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_MultLayer_SitSplitShuffle.pth",
                  "./cnn/saved_models/LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_MultLayer_SitSplitShuffle_test",
                  model=CNNFlowOnlyWithPooling(3, last_layer=True), dataset_class=DatasetOptFlo)

    process_video(video, "./cnn/saved_models/LeakyReLU_FramesSIAMESE_SitSplit.pth",
                      "./cnn/saved_models/LeakyReLU_FramesSIAMESE_SitSplit_test",
                      model=CnnSiamese(3), dataset_class=DatasetFrames)

    process_video(video, "./cnn/saved_models/LeakyReLU_FramesCONV_SitSplit.pth",
                      "./cnn/saved_models/LeakyReLU_FramesCONV_SitSplit_test",
                      model=CnnFramesConv(), dataset_class=DatasetFrames)

    process_video(video, "./cnn/saved_models/LeakyReLU_FramesOfCONV_SitSplit.pth",
                      "./cnn/saved_models/LeakyReLU_FramesOfCONV_SitSplit_test",
                      model=CnnFramesOfConv(), dataset_class=DatasetOptFlo2Frames)

    process_video(video, "./cnn/saved_models/SuperSuperSimple.pth",
                  "./cnn/saved_models/SuperSuperSimple_test",
                  model=CnnSuperSimple(), dataset_class=DatasetOptFlo)

txt1 = np.genfromtxt("./cnn/saved_models/LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_MultLayer_SitSplitShuffle_test.txt")
txt2 = np.genfromtxt("./cnn/saved_models/LeakyReLU_MixedSIAMESE_SitSplit_test.txt")
txt3 = np.genfromtxt("./cnn/saved_models/LeakyReLU_FramesSIAMESE_SitSplit_test.txt")
txt4 = np.genfromtxt("./cnn/saved_models/LeakyReLU_FramesCONV_SitSplit_test.txt")
txt5 = np.genfromtxt("./cnn/saved_models/LeakyReLU_FramesOfCONV_SitSplit_test.txt")
txt6 = np.genfromtxt("./cnn/saved_models/SuperSuperSimple_SitSplit_test.txt")

kernel_size = 50
kernel = np.ones(kernel_size) / kernel_size
txt1_convolved = np.convolve(txt1, kernel, mode='same')
txt2_convolved = np.convolve(txt2, kernel, mode='same')
txt3_convolved = np.convolve(txt3, kernel, mode='same')
txt4_convolved = np.convolve(txt4, kernel, mode='same')
txt5_convolved = np.convolve(txt5, kernel, mode='same')
txt6_convolved = np.convolve(txt6, kernel, mode='same')

#plt.plot(txt1, color="black", label="Normal (SitSplitShuffle)", zorder=0)
#plt.plot(txt2, color="gray", label="Mixed Siamese (SitSplitShuffle)", zorder=0)
#plt.plot(txt1_convolved, color="blue", label="Normal Smoothed (SitSplitShuffle)", zorder=1)
#plt.plot(txt2_convolved, color="blue", label="Mixed Siamese Smoothed (SitSplitShuffle)", zorder=1)
#plt.plot(txt3_convolved, color="blue", label="Frame Siamese Smoothed (SitSplitShuffle)", zorder=1)
#plt.plot(txt4_convolved, color="blue", label="Frames Conv Smoothed (SitSplitShuffle)", zorder=1)
#plt.plot(txt5_convolved, color="blue", label="FramesOf Conv Smoothed (SitSplitShuffle)", zorder=1)
plt.plot(txt6_convolved, color="blue", label="FramesOf Conv Smoothed (SitSplitShuffle)", zorder=1)

sits = np.array([360, 1100, 1600, 2140, 5400, 6180, 6640, 7400, 8160, 8860, 8920, 9660, 9780, 10420])

plt.vlines(x=sits, ymin=0, ymax=25)

plt.legend()
#plt.savefig("./cnn/saved_models/Videos/AllTest.eps")
#plt.savefig("./cnn/saved_models/Videos/AllTest.png")
#plt.vlines(x=0.8*20400, ymin=0, ymax=25, colors="red")
plt.show()
