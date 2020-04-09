import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

url = 'training'
dsturl = 'segmented/'
for filename in os.listdir(url + '/raw'):
    print(filename)
    # im = np.array(plt.imread(os.path.join(url + '/raw', filename)))
    # print(im.shape)
    # fig = plt.figure(figsize=(7, 4))
    # for i in range(int(row)):
    #     for j in range(int(col)):
    #         sub = im[i * SIZE : (i + 1) * SIZE, j * SIZE : (j + 1) * SIZE, :]
    #
    #         fig.add_subplot(row, col, int(col * i + j) + 1)
    #         plt.axis('off')
    #         plt.imshow(sub)
    #
    # plt.show()

    im = Image.open(os.path.join(url + '/raw', filename))
    width, height = im.size
    SIZE = 80
    row = height / SIZE
    col = width / SIZE

    rad = random.random()
    rad = str(int(rad * 1000))
    for i in range(int(row)):
        for j in range(int(col)):
            im1 = im.crop((j * SIZE, i * SIZE, (j + 1) * SIZE, (i + 1) * SIZE))
            im1.save(dsturl + filename + '_' + str(i) + '_' + str(j) + '-' + rad + '.png')
    print(filename + ' DONE.')