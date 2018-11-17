import imutils
import mp as mp
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
import skimage.morphology as mp
from skimage.transform import rescale

print("start___")

img_name = '20181107_195516.jpg'
img = cv2.imread(os.path.join('resources/PlanszaBezKolekIPionkow', img_name))
#img = rgb2gray(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def czarnaPizza(img):
    bounds = [[0, 3, 94], [110, 145, 220]]

    lower = np.array(bounds[0], dtype="uint8")
    upper = np.array(bounds[1], dtype="uint8")

    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    # output = rgb2gray(output)
    thresh, dst = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY)  # JESLI CHCESZ ZMIENIC KOLORY
                                                                    # THRESHOLDA (INVERTED)
                                                                    # DOPISZ _INV DO OSTATNIEGO ARGUMENTU
    dst=rgb2gray(dst)
    for i in range(10):
        dst = mp.dilation(dst)
        dst = mp.erosion(mp.dilation(dst))

    print(type(dst))
    # info=np.iinfo(dst.dtype)
    # dst = dst.astype(np.float64) / info.max
    # dst = 255*dst
    dst=dst.astype(np.uint8)
    print(dst.dtype)
    im2, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, 2)
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    print(area)
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    print(epsilon)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    print(cnt)
    print(cnt.shape)
    # hull = cv2.convexHull(cnt)
    print(cnt.shape)
    # small = cv2.resize(dst, (700, 700))
    # cv2.imshow("image", dst)
    # cv2.drawContours(dst, cnt, -1, (0,255,0), 3)
    # cv2.waitKey(0);
    io.imshow(dst, cmap='gray')
    plt.show()

    return output


wynik = czarnaPizza(img)