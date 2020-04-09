import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from os import listdir
from os.path import isfile, join
from skimage import transform
from PIL import Image

pathIn = '../signs/temporary/'
onlyfiles = [f for f in listdir(pathIn) if isfile(join(pathIn, f))]

print(onlyfiles)
i = 0
for file in onlyfiles:
    im = plt.imread(pathIn + file)
    im = im[:,:,0:3]
    im = transform.resize(im, (224, 224))
    matplotlib.image.imsave(pathIn + str(i) + '.jpg', im)
    i += 1

# for i in range(30, 190):
#     filename = pathIn + str(i) + '.png'
#     print(filename)
#
#     im = plt.imread(filename)
#
#     fig, ax = plt.subplots(1)
#     fig.set_size_inches(12, 7, forward=True)
#
#     ax.imshow(im)
#
#
#     plt.gca().set_axis_off()
#     plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
#                         hspace=0, wspace=0)
#     plt.margins(0, 0)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.savefig("testing/result/" + str(i) + ".png", bbox_inches='tight',
#                 pad_inches=0)
#     # plt.savefig("testing/result/test3.png", bbox_inches='tight',
#     #             pad_inches=0)
#     plt.cla()
#     plt.close(fig)
#
