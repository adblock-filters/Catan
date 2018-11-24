import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, filters, exposure, measure, segmentation
from skimage.morphology import disk, opening, dilation, square
from skimage.feature import match_template

from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray, gray2rgb
from skimage.filters.edges import convolve

# img_name = 'ex1.jpg'
# img = io.imread(os.path.join('resources/', img_name))

# draw_contours(img, contours_by_saturation(img, lightness_mask(img)))
# draw_contours(img, contours_by_saturation(img), contours_by_sobel(img))
# draw_contours(img, contours_by_sobel(img))

# himg = cv2.imread('resources/ex1.jpg',)
# cv2.cvtColor(himg, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(himg)


# load the image, clone it for output, and then convert it to grayscale
img = cv2.imread('resources/ex1.jpg', cv2.IMREAD_COLOR)
img2 = img

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#img = cv2.blur(img, (1, 1))
imghsv = img
h, s, v = cv2.split(img)
v=v*2
# h.fill(200)
# s.fill(150)
# v.fill(2)

imghsv = cv2.merge([h, s, v])

# imghsv = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)

# imghsv = 255 - imghsv
cv2.imwrite('resources/ex2.jpg', imghsv)

# imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2GRAY)
# mask = imghsv < 65
# imghsv[mask] = 255
# cv2.imwrite('resources/ex2.jpg', imghsv)

# contours = measure.find_contours(imghsv, 0.8)
# fig, ax, = plt.subplots()
# ax.imshow(imghsv, interpolation='nearest', cmap=plt.cm.gray)
#
# print(len(contours))
#
# for n, contour in enumerate(contours):
#     ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)
#
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()

# im2, contours, hierarchy=cv2.findContours(imghsv,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(imghsv, contours, -1, (0, 0, 255), 33)
#
#
# cv2.namedWindow('ex', cv2.WINDOW_NORMAL)
# cv2.imshow('ex', imghsv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
1. histogram flatten
https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
2. blur - averaging
https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
3. cut brighter (hue mask / satur mask)
4. find conturus by opencv - find equals / find with filed = image size/some_number
5. find circles with numbers - negativ - darkest circles

0. desert - find darkest element - contours
0. red house:
    hue ~-127 - darker / more blue than other
0. orange house:
    minimal hue - cut blue element
0. forest - negativ, maximal bright/light - white spaces
0. mountain - saturation max,  hue 80 - pink spaces / / saturation max, hue - 110 - green spaces
0. yellow fields - contrast max, saturation max, cyjan min(-100), hue -60
0. green fields - img = img[:,:,:]*2 - blue
0. find houses - max contrast, max saturation, gamma -50, cyjan -70, green 30, blue 100

"""

# cv2.imwrite('resources/ex2.jpg', himg)

print("__end")
