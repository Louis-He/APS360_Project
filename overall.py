import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage import filters
from skimage import measure
from skimage import transform
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F


# import cv2
#
# timeF = 2
# vc = cv2.VideoCapture('testing/202003271659_003331AA.mp4')
# c = 1
#
# if vc.isOpened():
#     rval, frame = vc.read()
# else:
#     rval = False
#
# index = 1
# while rval:
#     rval, frame = vc.read()
#     # if(c % timeF == 0):
#     cv2.imwrite('testing/frame/' + str(index) + '.jpg', frame)
#     index = index + 1
#     c = c + 1
#     cv2.waitKey(1)
#
#     if index == 600:
#         break
#
# vc.release()
#
# print("done")

MIN_INTERESTED = 20 # minimal interested area: 100 pixel
MARGIN = 0.05

def rgb2ycbcr(im):
    im = im * 255
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 127
    return np.uint8(ycbcr)

def getInterestedArea(im):
    modifyIm = im.copy()
    HSVIm = matplotlib.colors.rgb_to_hsv(modifyIm)
    HSV_V = HSVIm[:, :, 2]
    HSV_S = HSVIm[:, :, 1]
    HSV_S[HSV_S < 0.48] = 0
    HSV_S = filters.gaussian(HSV_S, sigma=5.5)
    blobs = (HSV_S > 15 * HSV_S.mean())

    modifyIm = im.copy()
    modifyIm_ycbcr = rgb2ycbcr(modifyIm)
    modifyIm_ycbcr = modifyIm_ycbcr / 255
    modifyIm_ycbcr = modifyIm_ycbcr[:, :, 2]
    modifyIm_ycbcr[modifyIm_ycbcr > 0.6] = 1
    modifyIm_ycbcr[modifyIm_ycbcr < 0.6] = 0

    combinedBlobs = blobs + modifyIm_ycbcr
    combinedBlobs[combinedBlobs >= 1] = 1
    all_labels = measure.label(combinedBlobs)
    blobs_labels = measure.label(combinedBlobs, background=0)
    maxLabel = np.max(blobs_labels)

    interesetedArea = []
    for i in range(1, maxLabel + 1):
        areaDetail = np.where(blobs_labels == i)
        area = len(areaDetail[0])

        if (area < MIN_INTERESTED):
            blobs_labels[blobs_labels == i] = 0
            continue
        # minrow, maxrow, mincol, maxcol
        areaLocation = [np.min(areaDetail[0]), np.max(areaDetail[0]), np.min(areaDetail[1]), np.max(areaDetail[1])]

        if (areaLocation[0] < len(combinedBlobs) * MARGIN or areaLocation[1] > len(combinedBlobs) - len(
                combinedBlobs) * MARGIN or
                areaLocation[2] < len(combinedBlobs[0]) * MARGIN or areaLocation[3] > len(combinedBlobs[0]) - len(
                    combinedBlobs[0]) * MARGIN):
            blobs_labels[blobs_labels == i] = 0
            continue

        interesetedArea.append(areaLocation)

    return interesetedArea

src = "testing/frame/"
modelsrc = "model_CNN_bs256_lr0.001_epoch5"

class CNN(nn.Module):
    # my CNN module goes from input data (3*224*224) to con1 (input_channels = 3, output_channels = 5, kernel_size = 5), the size becomes (5*220*220),
    # then it goes through an relu function and a maxpooling (kernal_size = 2, stride = 2), the size becomes (5*110*110),
    # then it goes through a second con2 (input_channels = 5, output_channels = 10, kernel_size = 5), the size becomes (10*106*106),
    # finally it goes through an relu function and a maxpooling (kernal_size = 2, stride = 2), the size becomes (10*53*53).
    # there are a total of 7 layers, including 2 convolution layers, 2 maxpooling layers, 2 fully connected layers, and 1 output layer

    def __init__(self):
        self.name = "CNN"
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 5)  # in_channels=3, out_chanels=5, kernel_size=5
        self.pool = nn.MaxPool2d(2, 2)  # kernel_size=2, stride=2
        self.conv2 = nn.Conv2d(5, 10, 5)  # in_channels=5, out_chanels=10, kernel_size=5
        self.fc1 = nn.Linear(10 * 53 * 53, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # relu activation function
        x = self.pool(F.relu(self.conv2(x)))  # relu activation function
        x = x.view(-1, 10 * 53 * 53)
        x = F.relu(self.fc1(x))  # relu activation function
        x = self.fc2(x)
        return x

model = CNN()
model.load_state_dict(torch.load(modelsrc, map_location=torch.device('cpu')))
model.eval()
criterion = nn.CrossEntropyLoss()
print('model load OK.')
MAX_HEIGHT = 495 # FIX height of the image no matter the size of the input


prior = None

for i in range(26, 600):
    print('Processing ' + str(i) + ".")
    imgSrc = src + str(i) + '.jpg'
    # imgSrc = src + "WechatIMG2127.jpeg"
    im = plt.imread(imgSrc)
    # im = im[110:950, 240:1680, :]
    oriIm = im.copy()
    # print(im.shape)

    scalingFactor = MAX_HEIGHT / im.shape[0]
    im = transform.resize(im, (MAX_HEIGHT, int(MAX_HEIGHT * im.shape[1] / im.shape[0])))

    # print(im.shape)

    AOI = getInterestedArea(im)
    # print(AOI)

    fig, ax = plt.subplots(1)
    fig.set_size_inches(12, 7, forward=True)
    ax.imshow(im)
    for areaLocation in AOI:
        oldAreaLoc = areaLocation.copy()

        # rect = patches.Rectangle((areaLocation[2], areaLocation[0]),
        #                          areaLocation[3] - areaLocation[2] + 1, areaLocation[1] - areaLocation[0] + 1,
        #                          linewidth=1, edgecolor='r', facecolor='none')
        for j in range(4):
            areaLocation[j] = int(areaLocation[j] / scalingFactor)

        # ycenter, xcenter
        center = (areaLocation[0] + areaLocation[1]) / 2, (areaLocation[2] + areaLocation[3]) / 2
        height = areaLocation[1] - areaLocation[0]
        width = areaLocation[3] - areaLocation[2]
        maxSize = max(height, width)

        if maxSize > 90:
            continue

        if maxSize < 50:
            maxSize = 55
        else:
            maxSize *= 1.1

        originalIm = oriIm[int(center[0] - maxSize / 2): int(center[0] + maxSize / 2),
                     int(center[1] - maxSize / 2): int(center[1] + maxSize / 2), :]

        originalIm = transform.resize(originalIm, (224, 224))
        originalIm = np.transpose(originalIm, axes=(2, 1, 0))
        output = model((torch.from_numpy(originalIm).float()).unsqueeze(0))

        # select index with maximum prediction score
        classes = ['other', 'regulatory', 'temporary', 'warning']
        colormaps = ['g', 'r', 'y', 'b']
        pred = output.max(1, keepdim=True)
        # print(pred)
        if pred[0] > 3:
            idx = classes[pred[1]]
            pltedgecolor = colormaps[pred[1]]
        else:
            idx = -1

        if idx != -1:
            rect = patches.Rectangle((oldAreaLoc[2], oldAreaLoc[0]),
                                                              oldAreaLoc[3] - oldAreaLoc[2] + 1, oldAreaLoc[1] - oldAreaLoc[0] + 1,
                                                              linewidth=2, edgecolor=pltedgecolor, facecolor='none')
            ax.add_patch(rect)
        else:
            rect = patches.Rectangle((oldAreaLoc[2], oldAreaLoc[0]),
                                     oldAreaLoc[3] - oldAreaLoc[2] + 1, oldAreaLoc[1] - oldAreaLoc[0] + 1,
                                     linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig("testing/result/" + str(i) + ".png", bbox_inches='tight',
    #             pad_inches=0)
    plt.savefig("testing/result/test2.png", bbox_inches='tight',
                pad_inches=0)
    plt.cla()
    plt.close(fig)

    break