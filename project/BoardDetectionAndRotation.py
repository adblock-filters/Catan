import imutils
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from skimage import data, io, filters, exposure, measure, segmentation
from skimage.filters import rank
from skimage.morphology import disk
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray, gray2rgb
from skimage.filters.edges import convolve

print("start___")

img_name = '20181107_195516.jpg'
img = io.imread(os.path.join('resources/PlanszaBezKolekIPionkow', img_name))
#img = rgb2gray(img)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def czarnaPizza(img):
    bounds = [[0, 3, 94], [110, 145, 220]]

    lower = np.array(bounds[0], dtype="uint8")
    upper = np.array(bounds[1], dtype="uint8")

    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    # output = rgb2gray(output)
    thresh, dst = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY_INV)
    io.imshow(dst)
    plt.show()

    return output


wynik = czarnaPizza(img)




