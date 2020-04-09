import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

pathIn = './testing/result/'
stopSign = 'stop.png'

for i in range(30, 190):
    filename = pathIn + str(i) + '.png'
    print(filename)

    im = plt.imread(filename)

    fig, ax = plt.subplots(1)
    fig.set_size_inches(12, 7, forward=True)

    ax.imshow(im)

    arr_lena = plt.imread(stopSign)
    imagebox = OffsetImage(arr_lena, zoom=0.8)
    ab = AnnotationBbox(imagebox, (50, 50))
    ax.add_artist(ab)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("testing/result/" + str(i) + ".png", bbox_inches='tight',
                pad_inches=0)
    # plt.savefig("testing/result/test3.png", bbox_inches='tight',
    #             pad_inches=0)
    plt.cla()
    plt.close(fig)

