import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from skimage import data, io, filters, exposure, measure, segmentation
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray, gray2rgb
from skimage.filters.edges import convolve

print("start___")

img_name = 'low_contrast.jpg'
img = io.imread(os.path.join('resources/', img_name))
img = rgb2gray(img)

contours = measure.find_contours(img, 0.1)
fig, ax = plt.subplots()
ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    # ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    # filtering
    if (
        contours[n-1][0, 1] - contour[0 , 1] < 5
        or
        contours[n - 1][0, 0] - contour[0, 0] < 5
    ):
        ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)


ax.set_xticks([])
ax.set_yticks([])
plt.show()

#io.imshow(img)
#plt.show()
