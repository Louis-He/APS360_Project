import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from scipy import signal
from scipy import ndimage
from skimage import filters
from skimage import measure
from skimage import transform
import cv2

def findGradient(imgArr):
    scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                        [-10+0j, 0+ 0j, +10 +0j],
                        [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
    grad = signal.convolve2d(imgArr, scharr, boundary='symm', mode='same')
    return grad
def rgb2ycbcr(im):
    im = im * 255
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 127
    return np.uint8(ycbcr)


MAX_HEIGHT = 495 # FIX height of the image no matter the size of the input
MIN_INTERESTED = 50 # minimal interested area: 100 pixel
MARGIN = 0.1

im = plt.imread("images/19.jpg")

# fig,ax = plt.subplots(1)
# fig.set_size_inches(18.5, 10.5, forward=True)
# ax.imshow(im)
# if im.shape[1] != MAX_HEIGHT:
scalingFactor = MAX_HEIGHT / im.shape[1]
im = transform.resize(im, (MAX_HEIGHT, int (MAX_HEIGHT * im.shape[1] / im.shape[0])))

modifyIm = im.copy()
HSVIm = matplotlib.colors.rgb_to_hsv(modifyIm)
HSV_V = HSVIm[:,:,2]
HSV_S = HSVIm[:,:,1]
HSV_S[HSV_S < 0.48] = 0
# plt.imshow(HSV_S, cmap='gray')
plt.imshow(HSV_S, cmap='gray')
plt.savefig('test_HSV.jpg')

HSV_S = filters.gaussian(HSV_S, sigma=5)

blobs = (HSV_S > 15 * HSV_S.mean())
# plt.imshow(blobs, cmap='gray')

modifyIm = im.copy()
print(np.max(modifyIm))

modifyIm_ycbcr = rgb2ycbcr(modifyIm)
modifyIm_ycbcr = modifyIm_ycbcr / 255
modifyIm_ycbcr = modifyIm_ycbcr[:,:,2]
modifyIm_ycbcr[modifyIm_ycbcr > 0.6] = 1
modifyIm_ycbcr[modifyIm_ycbcr < 0.6] = 0
plt.imshow(modifyIm_ycbcr, cmap='gray')
plt.savefig('test_YCbCr.jpg')

combinedBlobs = blobs + modifyIm_ycbcr
combinedBlobs[combinedBlobs>=1] = 1
# fig,ax = plt.subplots(1)
# fig.set_size_inches(18.5, 10.5, forward=True)
#
# ax.imshow(combinedBlobs, cmap='gray')

fig,ax = plt.subplots(1)
fig.set_size_inches(18.5, 10.5, forward=True)

ax.imshow(im)


all_labels = measure.label(combinedBlobs)
blobs_labels = measure.label(combinedBlobs, background=0)
maxLabel = np.max(blobs_labels)

interesetedArea = []
for i in range(1, maxLabel + 1):
    areaDetail = np.where(blobs_labels == i)
    area = len(areaDetail[0])

    if(area < MIN_INTERESTED):
        blobs_labels[blobs_labels == i] = 0
        continue
    # minrow, maxrow, mincol, maxcol
    areaLocation = [np.min(areaDetail[0]), np.max(areaDetail[0]), np.min(areaDetail[1]), np.max(areaDetail[1])]

    if(areaLocation[0] < len(combinedBlobs) * MARGIN or areaLocation[1] > len(combinedBlobs) - len(combinedBlobs) * MARGIN or
       areaLocation[2] < len(combinedBlobs[0]) * MARGIN or areaLocation[3] > len(combinedBlobs[0]) - len(combinedBlobs[0]) * MARGIN):
       blobs_labels[blobs_labels == i] = 0
       continue

    areaCandidate = (areaLocation[1] - areaLocation[0] + 1) * (areaLocation[3] - areaLocation[2] + 1)

    rect = patches.Rectangle((areaLocation[2], areaLocation[0]),
                             areaLocation[3] - areaLocation[2] + 1, areaLocation[1] - areaLocation[0] + 1,
                             linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    interesetedArea.append(areaLocation)

# plt.show()
plt.savefig('test.jpg')

for singleArea in interesetedArea:
    print(singleArea)