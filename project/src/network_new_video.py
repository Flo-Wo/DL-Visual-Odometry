from network_user import process_video
from cnn.cnn_frames_convolutional import CnnFramesConv
from data_loader import DatasetFrames, DatasetOptFlo1Frames, DatasetOptFlo
from cnn.cnn_siamese_frames_flow import CnnSiamese
import numpy as np
from matplotlib import pyplot as plt

from cnn.cnn_flow_only_with_pooling import CNNFlowOnlyWithPooling

prodVideos = False

video = "./data/raw/train.mp4"

if prodVideos:
    process_video(video, "./cnn/saved_models/NewSplitting/LeakyReLU_Frames_Conv.pth",
                  "./cnn/saved_models/Videos/LeakyReLU_Frames_Conv_train", model=CnnFramesConv(), dataset_class=DatasetFrames)

    process_video(video, "./cnn/saved_models/NewSplitting/LeakyReLU_SIAMESE.pth",
                  "./cnn/saved_models/Videos/LeakyReLU_SIAMESE_train", model=CnnSiamese(3), dataset_class=DatasetOptFlo1Frames)

    process_video(video, "./cnn/saved_models/NewSplitting/LeakyReLU_Frames_Conv_SitSplit.pth",
                  "./cnn/saved_models/Videos/LeakyReLU_Frames_Conv_SitSplit_train", model=CnnFramesConv(),
                  dataset_class=DatasetFrames)

    process_video(video, "./cnn/saved_models/NewSplitting/LeakyReLU_SIAMESE_SitSplit.pth",
                  "./cnn/saved_models/Videos/LeakyReLU_SIAMESE_SitSplit_test", model=CnnSiamese(3),
                  dataset_class=DatasetOptFlo1Frames)

    process_video(video, "./cnn/saved_models/NewSplitting/LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_new.pth",
                      "./cnn/saved_models/Videos/LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_new_train",
                      model=CNNFlowOnlyWithPooling(3), dataset_class=DatasetOptFlo)

txt1 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_SIAMESE_train.txt")
txt2 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_Frames_Conv_train.txt")
txt3 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_SIAMESE_SitSplit_test.txt")
txt4 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_Frames_Conv_SitSplit_train.txt")
txt5 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU15EpochsBatchNormMaxPoolingWithDropOut_train.txt")*1.66
txt53 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_train.txt")*1.4
txt51 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_MultLayer_SitSplitShuffle_train.txt")*1.85
txt52 = np.genfromtxt("./cnn/saved_models/Videos/LeakyReLU_25Epochs_BatchNorm_MaxPooling_WithDropout_MultLayer_SitSplit_train.txt")*1.85
txt6 = np.genfromtxt("./data/raw/train_label.txt")

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
txt51_convolved = np.convolve(txt51, kernel, mode='same')
txt52_convolved = np.convolve(txt52, kernel, mode='same')
txt53_convolved = np.convolve(txt53, kernel, mode='same')
#txt6_convolved = np.convolve(txt6, kernel, mode='same')

#dif = np.sqrt((txt5-txt6[1:])**2)

#plt.plot(dif)
#print(np.sqrt(np.sum(dif)/(20399)))

#plt.plot(txt1_convolved, color="black", label="Siamese Smoothed (100S)")
#plt.plot(txt2_convolved, color="gray", label="Conv. Smoothed (100S)")
#plt.plot(txt3_convolved, color="gray", label="Siamese Smoothed (SitS)")
#plt.plot(txt4_convolved, color="black", label="Conv. Smoothed (SitS)")
#plt.plot(txt5_convolved, color="blue", label="Classical (HardS)", zorder=1)
plt.plot(txt5_convolved, color="gray", ls="-", marker="None", label="15 (old)", zorder=-1)
plt.plot(txt51_convolved, color="black", ls="-", marker="None", label="25 (shuffle)", zorder=-1)
plt.plot(txt52_convolved, color="red", ls="-", marker="None", label="25 (no shuffle)", zorder=-1)
plt.plot(txt53_convolved, color="blue", ls="-", marker="None", label="25 (old)", zorder=-1)
plt.plot(txt6, color="green", label="Labels", zorder=0)

print(np.sqrt(np.sum((txt51-txt6[1:])**2)/len(txt6)))

plt.legend()
#plt.savefig("./cnn/saved_models/Videos/AllTest.eps")
#plt.savefig("./cnn/saved_models/Videos/AllTest.png")
#plt.vlines(x=0.8*20400, ymin=0, ymax=25, colors="red")
plt.show()
