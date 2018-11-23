#import imutils
import os

import cv2
import matplotlib.pyplot as plt
import mp as mp
import numpy as np
# from skimage.filters.edges import convolve
import skimage.morphology as mp
from skimage import io
from skimage.color import rgb2gray

print("start")
img_name = '20181107_201012.jpg'
img = io.imread(os.path.join('resources/PlanszaZKolkamiIPionkami', img_name))
#img = rgb2gray(img)

def BoardAssignment(img):
    #swapped = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  ##DO ZAKOMENTOWANIA JESLI
                                               ##OBRAZEK JEST ŁADOWANY PRZEZ io.imread
                                               ##(POTRZEBNE DLA cv2.imread)

    swapped =img                               #JEŚLI ŁADOWANE moudłem skimage
    bounds = [[0, 3, 94], [110, 170, 255]]

    lower = np.array(bounds[0], dtype="uint8")
    upper = np.array(bounds[1], dtype="uint8")

    mask = cv2.inRange(swapped, lower, upper)
    filtered = cv2.bitwise_and(swapped, swapped, mask=mask)

    thresh, dst = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY)  # JESLI CHCESZ ZMIENIC KOLORY
                                                                  # DOPISZ _INV DO OSTATNIEGO ARGUMENTU

    # for i in range(10):
    #  dst = mp.dilation(dst)
    # dst = mp.erosion(mp.dilation(dst))
    # info=np.iinfo(dst.dtype)
    # dst = dst.astype(np.float64) / info.max
    # dst = 255*dst
    dst=rgb2gray(dst)
    dst=dst.astype(np.uint8)
    im2, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, 1)
    longestContour = contours[0]
    for i in contours:
        if(i.shape[0]>longestContour.shape[0]):
            longestContour=i

    xmax=0
    ymax=0
    for row in longestContour:
        for pixel in row:
            if pixel[0]>xmax:
                xmax=pixel[0]
            if pixel[1] > ymax:
                ymax = pixel[1]
    xmin=xmax
    ymin=ymax
    for row in longestContour:
        for pixel in row:
            if pixel[0] < xmin:
                xmin = pixel[0]
            if pixel[1] < ymin:
                ymin = pixel[1]

    output=img[ymin:ymax, xmin:xmax]
    # cnt = longestContour
    # print(cnt)
    # print(cnt.shape)
    # print(cnt.shape[0])
    # print(cnt[0,:])
    # hull = cv2.convexHull(cnt)
    # print(cnt.shape)
    # small = cv2.resize(dst, (700, 700))
    # cv2.drawContours(swapped, longestContour, -1, color=(255,0,0), thickness=5)
    #cv2.imwrite("image", dst)
    #plt.imshow(dst, cmap='gray')
    # plt.imshow(swapped, cmap='gray')
    # plt.show()
    return output

wynik = BoardAssignment(img)
#wynik = cv2.cvtColor(wynik, cv2.COLOR_BGR2RGB)
plt.imshow(wynik)
plt.show()