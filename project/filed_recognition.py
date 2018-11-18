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

"""
WYKRYWANIE POL
0.1 uporzadkuj punkty w tablicy rosnaco, osobno dla kazdej z 4 tablic: S-N, W-E, SW-NE, NW-SE
0.2 wylicz x - dlugosc boku na podstawie stosunku do dlugosci obrazka
0.3 wylicz r na podstawie x - promien krotki 6kata
1. wykrywanie linii
1.1 dla kadego punktu sprawdx jego 2 nastepniki i jesli tworza linie, to dodaj ten punkt do tablicy
1.2 idz tak dlugo az roznica miedzy polozeniem pierwszego a ostatniego wyniesie x 
(wyliczany na podstawie stosunku dlugosci niebieskich bokow planszy)
1.3 narysuj najdluzsza mozliwa linie 
(1.4) lub narysuj linie przez cala plansze
2. tworzenie linii
2.1 ze zbioru punktow stworz linie kontutu o dlugosci x
3. wykrywanie naprzeciwlegej rwnolegej linii
3.1 jesli dla okreslonej przez y liczby punktow sa polozone rownolegle ok. y punkty, 
to WYKRYTO
4. rysowanie 6kata na podstawie 2 linii i ich odlegosci
4.1 wylicz dokladna odleglosc miedzy liniami = L
{?} 4.2 znajdz srodek tych linii (?)
4.3 narysuj 6kat
"""

print("start___")

img_name = 'low_contrast.jpg'
img = io.imread(os.path.join('resources/', img_name))
img2 = img
img3 = img


def lightness_mask(img_rgb, lightness=0.45, gamma=3, remove_disk=2, rejoin=8):
    # params:
    lightness_param = lightness
    gamma_param = gamma
    remove_disk_param = remove_disk
    rejoin_square_param = rejoin

    # enhance contrast
    img = exposure.adjust_gamma(img_rgb, gamma_param, 1)

    # split to h s v
    img = rgb2hsv(img)
    h = img[:, :, 0]
    s = img[:, :, 1]
    v = img[:, :, 2]

    # make lightness mask
    mask = (v > lightness_param).astype(np.uint8)

    # remove small regions from mask
    disk_elem = disk(remove_disk_param)
    opened = opening(mask, selem=disk_elem)

    # rejoin colored pionts
    square_elem = square(rejoin_square_param)
    dilated = dilation(opened, selem=square_elem)

    io.imshow(dilated)
    plt.show()

    return dilated.astype(bool)


def saturation_mask(img_rgb, saturation=0.75, remove=0, rejoin=5):
    # params:
    saturation_param = saturation
    remove_disk_param = remove
    rejoin_square_param = rejoin

    # split to h s v
    img = rgb2hsv(img_rgb)
    h = img[:, :, 0]
    s = img[:, :, 1]
    v = img[:, :, 2]
    # create mask
    mask = (s > saturation_param).astype(np.uint8)

    # remove small regions from mask
    disk_elem = disk(remove_disk_param)
    opened = opening(mask, selem=disk_elem)

    # rejoin colored pionts
    square_elem = square(rejoin_square_param)
    dilated = dilation(opened, selem=square_elem)

    io.imshow(dilated)
    plt.show()

    return dilated.astype(bool)


def contours_by_saturation(img, func=saturation_mask(img), find_param=0.05):
    # params:
    contour_find_param = find_param
    # set mask
    img[func, :] = 0;
    img = rgb2gray(img)
    # find contours
    contours = measure.find_contours(img, contour_find_param)
    # select long contours
    long_contours = []
    for contour in contours:
        if len(contour) > 900:
            long_contours.append(contour)

    return long_contours


def draw_contours(img, func):
    contours = func

    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def sobel_mask(img, mask=0.1, gamma=3, gauss=6):
    # params:
    gamma_param = gamma
    mask_param = mask
    gauss_param = gauss

    img = exposure.adjust_gamma(img, gamma_param, 1)
    img = filters.sobel(rgb2gray(img))
    img = gray2rgb(img)

    mask_dark = img <= mask_param
    mask_light = img > mask_param
    img[mask_dark] = 1;
    img[mask_light] = 0;

    img = filters.gaussian(img, gauss_param)
    img = exposure.adjust_gamma(img, gamma_param, 1)

    io.imshow(img)
    plt.show()

    return img


def contours_by_sobel(img_rgb, find_param=0.7, sobel=0.08, len=1000):
    # params:
    contour_find_param = find_param
    contour_len = len
    sobel_param = sobel
    # find contours
    img = rgb2gray(sobel_mask(img_rgb, sobel_param))
    contours = measure.find_contours(img, contour_find_param)
    # select long contours
    long_contours = []
    for contour in contours:
        if len(contour) > contour_len:
            long_contours.append(contour)

    return long_contours



# draw_contours(img, contours_by_saturation(img))
# draw_contours(img, contours_by_sobel(img))
image = io.imread(os.path.join('resources/', 'fortemplate2.png'))
field = io.imread(os.path.join('resources/', 'template.png'))
image = rgb2gray(image)
field = rgb2gray(field)



