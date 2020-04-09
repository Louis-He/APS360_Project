import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset
import os

class CNN(nn.Module):
    # my CNN module goes from input data (3*224*224) to con1 (input_channels = 3, output_channels = 5, kernel_size = 5), the size becomes (5*220*220),
    # then it goes through an relu function and a maxpooling (kernal_size = 2, stride = 2), the size becomes (5*110*110),
    # then it goes through a second con2 (input_channels = 5, output_channels = 10, kernel_size = 5), the size becomes (10*106*106),
    # finally it goes through an relu function and a maxpooling (kernal_size = 2, stride = 2), the size becomes (10*53*53).
    # there are a total of 7 layers, including 2 convolution layers, 2 maxpooling layers, 2 fully connected layers, and 1 output layer

    def __init__(self):
        self.name = "CNN"
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 10, stride=1, padding=5)  # in_channels=3, out_chanels=5, kernel_size=10
        self.conv2 = nn.Conv2d(10, 1, 10, stride=1, padding=4)  # in_channels=3, out_chanels=5, kernel_size=10
        # self.conv3 = nn.Conv2d(5, 1, 10, stride=1, padding=5)  # in_channels=3, out_chanels=5, kernel_size=10

    def forward(self, x):
        x = F.relu(self.conv1(x))  # relu activation function
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        # x = self.conv3(x)
        # print(x.shape)
        return x


def train_small(model, raw, annotated, batch_size=4, learning_rate=0.001, num_epochs=10):
    # train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    iters, losses, train_acc = [], [], []

    # training
    epoch = 0  # the number of iterations
    for epoch in range(num_epochs):
        lossTot = 0
        for idx in range(len(raw)):

            out = model(torch.from_numpy(raw[idx]).float())  # forward pass
            loss = criterion(out, torch.from_numpy(annotated[idx]).float())  # compute the total loss
            lossTot += loss
            loss.backward()  # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter
            optimizer.zero_grad()  # a clean up step for PyTorch

        # save the current training information
        iters.append(epoch)
        losses.append(float(lossTot) / batch_size)  # compute *average* loss
        # train_acc.append(get_accuracy_small(model, small_dataloader))  # compute training accuracy
        print("epoch number ", epoch + 1, "loss: ", losses[epoch])
        # if get_accuracy_small(model, small_dataloader) == 1:
        #     break

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    # plt.title("Training Curve")
    # plt.plot(iters, train_acc, label="Train")
    # plt.xlabel("Iterations")
    # plt.ylabel("Training Accuracy")
    # plt.legend(loc='best')
    # plt.show()

    # print("Final Training Accuracy: {}".format(train_acc[-1]))


def initializeData(url, batchSize = 4):
    rawData = []
    AnnotatedData = []
    imageSize = []

    i = 4
    idx = -1
    for filename in os.listdir(url + '/processed'):
        print(filename)
        im = np.array(plt.imread(os.path.join(url + '/raw', filename)))
        im = np.transpose(im, axes=(2, 1, 0))
        imageSize = im.shape

        annotatedIm = np.array(plt.imread(os.path.join(url + '/processed', filename)))
        annotatedIm = np.transpose(annotatedIm, axes=(1, 0, 2))
        annotatedImFlag = np.zeros((1, int(imageSize[1]), int(imageSize[2])))

        for i in range(imageSize[1]):
            for j in range(imageSize[2]):
                if annotatedIm[i][j] == [255, 255, 255]:
                    annotatedImFlag[0][i][j] = 1
                else:
                    annotatedImFlag[0][i][j] = 0

        if i < batchSize:
            # rawData[idx].append(im)
            rawData[idx][i] = im
            AnnotatedData[idx][i] = annotatedImFlag
            i += 1
        else:
            idx += 1
            i = 0
            rawData.append(np.empty([batchSize, im.shape[0], im.shape[1], im.shape[2]]))
            rawData[idx][i] = im

            AnnotatedData.append(np.empty([batchSize, annotatedImFlag.shape[0], annotatedImFlag.shape[1], annotatedImFlag.shape[2]]))
            AnnotatedData[idx][i] = annotatedImFlag

    print(imageSize)
    while i != 4:
        rawData[idx][i] = np.zeros([imageSize[0], imageSize[1], imageSize[2]])
        AnnotatedData[idx][i] = np.zeros([1, imageSize[1], imageSize[2]])
        i += 1

    return rawData, AnnotatedData

raw, annotated = np.array(initializeData('training'))
print(raw.shape, annotated.shape)
model = CNN()
train_small(model, raw, annotated)

