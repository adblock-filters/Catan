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
#from skimage.filters.edges import convolve


img_name = 'middle_contrast.jpg'

img = cv2.imread('resources/' + img_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, 1)
image, contours, h = cv2.findContours(thresh, 1, 2)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    print(len(approx))
    if len(approx) == 5:
        print("pentagon")
        cv2.drawContours(img, [cnt], 0, 255, -1)
    elif len(approx) == 3:
        print("triangle")
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), -1)
    elif len(approx) == 4:
        print("square")
        cv2.drawContours(img, [cnt], 0, (0, 0, 255), -1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()