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

# =====================================
# sample programs

contours_points = []
for n, cont in enumerate(contours):
    for i in cont:
        contours_points.append(i)

contours_points = np.asarray(contours_points)
contours_points.sort()

len_contours_points = len(contours_points)
linear_contours_points = []
for n, point in enumerate(contours_points):
    if (n + 2 < len_contours_points):
        point1 = contours_points[n + 1]
        point2 = contours_points[n + 2]

        if (point[0] == point1[0] == point2[0]):
            linear_contours_points.append(point)
        elif (point[0] == point1[0] or point[0] == point2[0]):
            pass
        else:
            if (((point1[1] - point[1]) / (point1[0] - point[0]))
                    -
                    ((point2[1] - point[1]) / (point2[0] - point[0]))
                    < 2
            ):
                linear_contours_points.append(point)

print(len(linear_contours_points))

# ---

len_contours_points = len(contours)
linear_contours = []
for i in range(len_contours_points): linear_contours.append([])
item = 0
for n, cont1 in enumerate(contours):
    if n + 1 < len_contours_points:
        c1x1 = cont1[0][0]
        c1y1 = cont1[0][1]
        c1x2 = cont1[int(len(cont1) / 2)][0]
        c1y2 = cont1[int(len(cont1) / 2)][1]

        c2x1 = contours[n + 1][0][0]
        c2y1 = contours[n + 1][0][1]
        c2x2 = contours[n + 1][int(len(contours[n + 1]) / 2)][0]
        c2y2 = contours[n + 1][int(len(contours[n + 1]) / 2)][1]

        linear_contours[item].append(cont1)
        if c1x2 == c1x1 or c2x2 == c1x1:
            linear_contours[item].append(contours[n + 1])
        else:
            if (((c1y2 - c1y1) / (c1x2 - c1x1))
                    -
                    ((c2y2 - c1y1) / (c2x2 - c1x1))
                    == 0
            ):
                linear_contours[item].append(contours[n + 1])
            else:
                item = n

linear_contours_numbers = []
for n, i in enumerate(linear_contours):
    if (len(i) > 2): linear_contours_numbers.append(n)

# ---