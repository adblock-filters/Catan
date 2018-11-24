import os
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, filters, exposure, measure, segmentation, draw, transform
from skimage.morphology import disk, opening, dilation, square
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray, gray2rgb
from board import BoardAssignment


def polygon_area(corners):
    """
    by Shoelace formula --- https://en.wikipedia.org/wiki/Shoelace_formula
    :param corners: array of coordinates of polygon
    :return: polygon area
    """
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def find_n_greatest_areas(contours, n_times=3):
    """
    :param contours: array of arrays of contours coordinates
    :param n_times: number of contours to find
    :return: N greatest contours (greatest areas)
    """
    cont_greatest_areas = []
    for i in range(n_times):
        cont_greatest_areas.append([0, 0])

    for n, contour in enumerate(contours):
        cont_area = polygon_area(contour)
        if cont_greatest_areas[0][0] < cont_area:
            del cont_greatest_areas[0]
            cont_greatest_areas.append([cont_area, n])
            cont_greatest_areas.sort()

    greatest_contours = []
    for i in range(n_times):
        greatest_contours.append(contours[cont_greatest_areas[i][1]])

    return greatest_contours


def check_if_circle(contours):
    """
    check by calculating std of distance between centroid and each contour point
    :param contours: array of arrays of contours coordinates
    :return: the most rounded contours
    """

    circles = []
    for n, contour in enumerate(contours):
        centroid = contour.mean(axis=0)
        distances = []
        for i in contour: distances.append(math.hypot(centroid[0] - i[0], centroid[1] - i[1]))
        if np.std(distances) < 16 and 200 < len(contour) < 1200:
            circles.append(contour)

    return circles


def create_mask(img_rgb, type='v', equation='>', saturation=0.75, remove=0, rejoin=5, show_mask=False):
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
    if equation == '>':
        if type == 'h':
            mask = (h > saturation_param).astype(np.uint8)
        elif type == 's':
            mask = (s > saturation_param).astype(np.uint8)
        else:
            mask = (v > saturation_param).astype(np.uint8)

    if equation == '<':
        if type == 'h':
            mask = (h < saturation_param).astype(np.uint8)
        elif type == 's':
            mask = (s < saturation_param).astype(np.uint8)
        else:
            mask = (v < saturation_param).astype(np.uint8)

    # remove small regions from mask
    disk_elem = disk(remove_disk_param)
    opened = opening(mask, selem=disk_elem)

    # rejoin colored points
    square_elem = square(rejoin_square_param)
    dilated = dilation(opened, selem=square_elem)

    if show_mask:
        io.imshow(dilated)
        plt.show()

    return dilated


# =========== HOUSES ============
"""
TODO:
- ( fix white houses )
"""


# FINISHED
def detect_orange_houses(img):
    func = create_mask(img, 's', '>', 0.8, 2, 10)

    contours = measure.find_contours(func, 0.05)
    new_contours = find_n_greatest_areas(contours, n_times=3)

    # ======= REMOVE =======
    fig, ax, = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(new_contours):
        ax.plot(contour[:, 1], contour[:, 0], 'orange', linewidth=3)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    # ======= REMOVE =======
    return new_contours


# FINISHED
def detect_blue_orange_houses(img):
    func = create_mask(img, 's', '>', 0.8, 2, 10)

    contours = measure.find_contours(func, 0.05)
    new_contours = find_n_greatest_areas(contours, n_times=3)

    orange_contours = []
    detected_extra_orange = True
    for n, contour in enumerate(new_contours):
        # find centroid of contour
        centroid = contour.mean(axis=0)
        # draw circle r=30px to check color of field
        rr, cc = draw.circle(centroid[0], centroid[1], 30, img.shape)
        # draw circle r=100px to overwrite field's color
        rr2, cc2 = draw.circle(centroid[0], centroid[1], 100, img.shape)
        # find color (h from hsv) and check if is orange or blue
        imghsv = rgb2hsv(img)
        h = imghsv[rr, cc, 0]
        h = np.interp(h, [0, 1], [0, 255])
        # if field is orange
        if h.mean() < 70:
            detected_extra_orange = False
        # or if field is blue
        else:
            # overwrite field's color by drawing r=100px circle
            img[rr2, cc2, :] = (1, 1, 1)
        # print(detected_extra_orange)

    if detected_extra_orange:
        orange_contours = detect_orange_houses(img)
        return [orange_contours, new_contours]
    else:
        return [orange_contours]


# FINISHED
def detect_red_houses(img):
    img = exposure.adjust_gamma(img, 2, 1)
    func = create_mask(img, 'h', '>', 0.95, 2, 10).astype(bool)

    contours = measure.find_contours(func, 0.05)
    new_contours = find_n_greatest_areas(contours, n_times=3)

    return new_contours


# FINISHED BUT WORKING BAD
def detect_white_houses(img):
    img = exposure.adjust_gamma(img, 2, 1)
    func = create_mask(img, 's', '<', 0.05, 0, 1).astype(bool)

    contours = measure.find_contours(func, 0.05)
    new_contours = find_n_greatest_areas(contours, n_times=2)

    # # ======= REMOVE =======
    # fig, ax, = plt.subplots()
    # ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    # for n, contour in enumerate(new_contours):
    #     ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=3)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()
    # # ======= REMOVE =======
    return new_contours


# FINISHED
def detect_black_joker(img):
    img = exposure.adjust_gamma(img, 2, 1)
    func = create_mask(img, 'v', '>', 0.2, 0, 3).astype(bool)

    contours = measure.find_contours(func, 0.05)
    new_contours = find_n_greatest_areas(contours, n_times=1)

    return new_contours


# =========== FIELDS ============
"""
TODO;
- fix mountain
- find grass --> extra saturation
- find grain --> lightness without desert and mountains
- find clay --> saturation + red color
"""


# FINISHED
def detect_circle_counters(img):
    img = exposure.equalize_hist(img)
    img_black = img

    func = create_mask(img, 's', '>', 0.4, 0, 4).astype(bool)
    img_black[func, :] = 0
    contours = measure.find_contours(rgb2gray(img_black), 0.2)

    circle_contours = check_if_circle(contours)
    circle_contours = find_n_greatest_areas(circle_contours, n_times=15)

    return circle_contours

# TODO
def detect_mountain_field(img):
    # when image around circle is '0' -> this is mountain filed
    func = create_mask(img, 's', '<', 0.25, 3, 1).astype(bool)
    img[func, :] = 0

    io.imshow(img)
    plt.show()


# FINISHED
def detect_forest_field(img):
    # when mean of image around circle is > '0' / < '255' -> this is forest filed
    # could also return (255-img) - may be easier to detect
    func = create_mask(img, 'v', '>', 0.55, 0, 2).astype(bool)
    img[func, :] = 0
    # additional color mask
    func = create_mask(img, 'h', '<', 0.1, 0, 1).astype(bool)
    img[func, :] = 0

    io.imshow(img)
    plt.show()






# -------START
names = ['ex1', 'ex2', 'ex3', 'ex4', 'ex5', 'ex6', 'ex71', 'ex72']
for name in names:
    img_name = name+'.jpg'
    img = io.imread(os.path.join('resources/', img_name))
    img_board = BoardAssignment(img)
    detect_circle_counters(img_board)
# -------END


# func = create_mask(img, 'h', '>', 0.95, 2, 10).astype(bool)
# # v, 0.85, 2, 4 - grain field
# img[func, :] = 0

# h = img[:, :, 0]
# s = img[:, :, 1]
# v = img[:, :, 2]
# h *= 2
# h = np.clip(h, 0, 255)
# func = create_mask(img, 'h', '<', 0.1, 0, 1).astype(bool)
# img[func, :] = 0;

# io.imsave('resources/orangehouses1.jpg', img)

# img2 = np.dstack((h, s, v))
# img2 = hsv2rgb(img2)
