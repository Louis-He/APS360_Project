from torch import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math


class CNN(nn.Module):

    def __init__(self):
        self.name = "CNN"
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)  # in_channels=3, out_chanels=5, kernel_size=5
        self.pool = nn.MaxPool2d(2, 2)  # kernel_size=2, stride=2
        self.conv2 = nn.Conv2d(10, 20, 5)  # in_channels=5, out_chanels=10, kernel_size=5
        self.fc1 = nn.Linear(20 * 17 * 17, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # relu activation function
        x = self.pool(F.relu(self.conv2(x)))  # relu activation function
        x = x.view(-1, 20 * 17 * 17)
        x = F.relu(self.fc1(x))  # relu activation function
        x = self.fc2(x)
        x = x.squeeze(1)
        return x

PATH = "model_CNN_bs64_lr0.0005_epoch29"
model = torch.load(PATH)
model.eval()

# imgURL = "images/9_21.png"
imgURL = "images/19_test.jpg"
# imgURL = "images/x_9.png"
im = plt.imread(imgURL)
print(im.shape)
im = np.transpose(im, axes=(2, 0, 1))
print(im.shape)
SIZE = 80

heightLen = (im.shape[1]) / SIZE
widthLen = (im.shape[2]) / SIZE
imgList = []
for i in range(0, int(heightLen) * 2 - 1):
    for j in range(0, int(widthLen) * 2 - 1):
        imgList.append(im[:, int(i * SIZE / 2) : int(i * SIZE / 2) + SIZE, int(j * SIZE / 2) : int(j * SIZE / 2) + SIZE] / 255)

imgs = torch.from_numpy(np.array(imgList)).float()
# print(imgs)
output = model(imgs)
print(output)

#
fig, ax = plt.subplots(1)
fig.set_size_inches(4, 4, forward=True)
for i in range(0, len(output)):
    if i == 300:
        row = math.floor(i / (int(widthLen) * 2 - 1))
        col = i - row * (int(widthLen) * 2 - 1)
        im = imgList[i]
        im = np.transpose(im, axes=(1, 2, 0))
        ax.imshow(im)
        plt.show()
        # plt.savefig('tmp/' + str(row)+'_'+str(col)+'.png')
        ax.clear()
        print(i, row, col, output[i])
    if output[i] > 0:
        row = math.floor(i / (int(widthLen) * 2 - 1))
        col = i - row * (int(widthLen) * 2 - 1)
        print(i, row, col, output[i])