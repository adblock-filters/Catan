import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from board import BoardAssignment
from skimage import io, exposure, measure, draw, filters
from skimage.color import rgb2hsv, rgb2gray
from skimage.morphology import disk, opening, dilation, square, closing, thin


# =========== OTHERS ============


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
    calculate area of contour and check if should be in N-greatest contours array
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


def check_if_circle(contours, std=16, lenmin=200, lenmax=1200):
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
        if np.std(distances) < std and lenmin < len(contour) < lenmax:
            circles.append(contour)

    return circles


def create_mask(img_rgb, hsv_type='v', equation='>', saturation=0.75, remove=0, rejoin=5, show_mask=False):
    # split to h s v
    img = rgb2hsv(img_rgb)
    h = img[:, :, 0]
    s = img[:, :, 1]
    v = img[:, :, 2]

    # create mask
    if equation == '>':
        if hsv_type == 'h':
            mask = (h > saturation).astype(np.uint8)
        elif hsv_type == 's':
            mask = (s > saturation).astype(np.uint8)
        else:
            mask = (v > saturation).astype(np.uint8)

    else:
        if hsv_type == 'h':
            mask = (h < saturation).astype(np.uint8)
        elif hsv_type == 's':
            mask = (s < saturation).astype(np.uint8)
        else:
            mask = (v < saturation).astype(np.uint8)

    # remove small regions from mask
    disk_elem = disk(remove)
    opened = opening(mask, selem=disk_elem)

    # rejoin colored points
    square_elem = square(rejoin)
    dilated = dilation(opened, selem=square_elem)

    if show_mask:
        io.imshow(dilated)
        plt.show()

    return dilated


def draw_contours(img, contours, color='r', width=3):
    fig, ax, = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], color, width)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


# =========== COUNTERS ============

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


# FINISHED
def detect_orange_houses(img):
    func = create_mask(img, 's', '>', 0.8, 2, 10)

    contours = measure.find_contours(func, 0.05)
    new_contours = find_n_greatest_areas(contours, n_times=3)

    return new_contours


# TODO FIX ORANGE
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

    if detected_extra_orange:
        orange_contours = detect_orange_houses(img)
        for item in orange_contours:
            new_contours.append(item)
        return (new_contours, True)
    else:
        return (new_contours, False)


# FINISHED
def detect_red_houses(img):
    img = exposure.adjust_gamma(img, 2, 1)
    func = create_mask(img, 'h', '>', 0.95, 2, 10).astype(bool)

    contours = measure.find_contours(func, 0.05)
    new_contours = find_n_greatest_areas(contours, n_times=3)

    return new_contours


# TODO: FIX PARAMS
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


# TODO: CHECK WITH ORANGE DETECTION
def draw_counters_and_circles(img, img_origin, white=True, red=True, black=True, orange=True, circle=True, fields=True):
    fig, ax, = plt.subplots(1, 2)
    img_source = img.copy()
    # original image
    ax[0].imshow(img_origin, interpolation='nearest', cmap=plt.cm.gray)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # draw white houses
    if white:
        for n, contour in enumerate(detect_white_houses(img_source)):
            centroid = np.mean(contour, axis=0)
            rr, cc = draw.circle(float(centroid[0]), float(centroid[1]), 90, img_source.shape)
            img[rr, cc, :] = (255, 255, 255)

    # draw red houses
    if red:
        for n, contour in enumerate(detect_red_houses(img_source)):
            centroid = np.mean(contour, axis=0)
            rr, cc = draw.circle(float(centroid[0]), float(centroid[1]), 90, img_source.shape)
            img[rr, cc, :] = (255, 20, 20)

    # draw black joker
    if black:
        for n, contour in enumerate(detect_black_joker(img_source)):
            centroid = np.mean(contour, axis=0)
            rr, cc = draw.circle(float(centroid[0]), float(centroid[1]), 90, img_source.shape)
            img[rr, cc, :] = (0, 0, 0)
            cv2.putText(img, 'PUSTYNIA', (int(centroid[1]), int(centroid[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)

    # draw orange and blue houses
    if orange:
        blue_and_orange = detect_blue_orange_houses(img_source)
        if blue_and_orange[1]:
            for n, contour in enumerate(blue_and_orange[0]):
                centroid = np.mean(contour, axis=0)
                rr, cc = draw.circle(float(centroid[0]), float(centroid[1]), 90, img_source.shape)
                if n > 2:
                    img[rr, cc, :] = (250, 178, 53)
                else:
                    img[rr, cc, :] = (62, 120, 250)
        else:
            for n, contour in enumerate(blue_and_orange[0]):
                centroid = np.mean(contour, axis=0)
                rr, cc = draw.circle(float(centroid[0]), float(centroid[1]), 90, img_source.shape)
                img[rr, cc, :] = (250, 178, 53)

    if fields:
        for n, contour in enumerate(detect_mountain_field(img_source)):
            centroid = np.mean(contour, axis=0)
            cv2.putText(
                img, 'GORY', (int(centroid[1]), int(centroid[0])), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
        for n, contour in enumerate(detect_clay_field(img_source)):
            centroid = np.mean(contour, axis=0)
            cv2.putText(
                img, 'KOPALNIA', (int(centroid[1]), int(centroid[0])), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
        for n, contour in enumerate(detect_grass_field(img_source)):
            centroid = np.mean(contour, axis=0)
            cv2.putText(
                img, 'TRAWA', (int(centroid[1]), int(centroid[0])), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
        for n, contour in enumerate(detect_forest_field(img_source)):
            centroid = np.mean(contour, axis=0)
            cv2.putText(
                img, 'LAS', (int(centroid[1]), int(centroid[0])), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)


    # show image on ax
    ax[1].imshow(img, interpolation='nearest', cmap=plt.cm.gray)

    # draw circle counters
    if circle:
        for n, contour in enumerate(detect_circle_counters(img_source)):
            ax[1].plot(contour[:, 1], contour[:, 0], 'navy', linewidth=3)
        for n, contour in enumerate(create_mountain_circles(img_source)):
            ax[1].plot(contour[:, 1], contour[:, 0], 'navy', linewidth=3)

    # show plot
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.show()


# =========== FIELDS ============


# FINISHED
def detect_mountain_field(img):
    # when image around circle is '0' -> this is mountain filed
    img_source = img
    img = exposure.equalize_hist(img)
    func = create_mask(img, 'h', '<', 0.55, 5, 5).astype(bool)
    img[func, :] = 255
    func = create_mask(img, 'h', '>', 0.8, 0, 5).astype(bool)
    img[func, :] = 255
    img = 255 - img

    contours = measure.find_contours(rgb2gray(img), 0.2)

    new_contours = check_if_circle(contours, 100, 1000, 4000)
    contours = find_n_greatest_areas(new_contours, n_times=3)

    # draw_contours(img_source, contours)

    return contours


# FINISHED
def create_mountain_circles(img):
    contours = detect_mountain_field(img)
    img2 = img - img
    for contour in contours:
        centroid = np.mean(contour, axis=0)
        rr, cc, = draw.circle(int(centroid[0]), int(centroid[1]), 70, img.shape)
        img2[rr, cc] = (255, 255, 255)

    new_contours = measure.find_contours(rgb2gray(img2), 0.2)
    # draw_contours(img, new_contours, 'navy')

    return new_contours


# FINISHED
def detect_forest_field(img2):#, circles):
    # when mean of image around circle is > '0' / < '255' -> this is forest filed
    # could also return (255-img) - may be easier to detect
    img = img2.copy()
    func = create_mask(img, 'v', '>', 0.55, 0, 2).astype(bool)
    img[func, :] = 0
    # additional color mask
    func = create_mask(img, 'h', '<', 0.1, 0, 1).astype(bool)
    img[func, :] = 0

    greenmask = img < 100
    img[greenmask] = 0

    greenmask = img > 100
    img[greenmask] = 255

    img = dilation(img)

    contours = measure.find_contours(rgb2gray(img), 0.1)

    new_contours = check_if_circle(contours, 50, 300, 4000)
    new_contours = find_n_greatest_areas(new_contours, n_times=4)
    #draw_contours(img_source, contours)

    return new_contours

    # forest_fields = []
    # for circle in circles:
    #     centroid = np.mean(circle, axis=0)
    #     rr, cc = draw.circle_perimeter(int(centroid[0]), int(centroid[1]), 200)
    #     # img[rr, cc, :] = (0, 0, 0)
    #     imghsv = rgb2hsv(img)
    #     h = imghsv[rr, cc, 0]
    #     h = np.interp(h, [0, 1], [0, 255])
    #     if h.mean() > 10:
    #         forest_fields.append(circle)
    # return forest_fields



# TODO GRAIN
def detect_grain_field(img):
    img_source = img
    img = exposure.equalize_hist(img)
    func = create_mask(img, 'h', '>', 0.15, 1, 1).astype(bool)
    img[func, :] = 0
    io.imshow(img)
    plt.show()

    # func = create_mask(img, 'v', '<', 0.8, 1, 5).astype(bool)
    io.imshow(img)
    plt.show()

    # grain_field = []
    # for circle in circles:
    #     centroid = np.mean(circle, axis=0)
    #     rr, cc = draw.circle_perimeter(int(centroid[0]), int(centroid[1]), 150, shape=img.shape)
    #
    #     imghsv = rgb2hsv(img)
    #     h = imghsv[rr, cc, 0]
    #     v = imghsv[rr, cc, 2]
    #     h = np.interp(h, [0, 1], [0, 255])
    #     v = np.interp(v, [0, 1], [0, 255])
    #
    #     mean_h = h.mean()
    #     if mean_h > 0:
    #         grain_field.append([v.mean(), circle])
    #
    # longest = []
    # if len(grain_field) > 4:
    #     newlist = sorted(grain_field, key=itemgetter(0))
    #     for i in range(4):
    #         longest.append(newlist[-(i + 1)][1])
    # else:
    #     for item in grain_field:
    #         longest.append(item[1])
    #
    # draw_contours(img_source, longest)
    # return longest


# FINISHED
def detect_clay_field(img):
    img_source = img
    img = exposure.equalize_hist(img)
    func = create_mask(img, 'h', '>', 0.08, 5, 5).astype(bool)
    img[func, :] = 255
    img = 255 - img

    contours = measure.find_contours(rgb2gray(img), 0.2)
    new_contours = check_if_circle(contours, 100, 1500, 4000)
    new_contours = find_n_greatest_areas(new_contours, n_times=3)
    # draw_contours(img_source, new_contours, 'navy')
    # io.imshow(img)
    # plt.show()
    return new_contours


# FINISHED
def detect_grass_field(img):
    # when image around circle is '0' -> this is mountain filed
    img_source = img
    img = exposure.equalize_hist(img)
    img = exposure.adjust_gamma(img, 2, 1)
    func = create_mask(img, 'h', '<', 0.25, 1, 1).astype(bool)
    img[func, :] = 0
    func = create_mask(img, 'h', '>', 0.35, 0, 1).astype(bool)
    img[func, :] = 0
    func = create_mask(img, 'v', '<', 0.4, 0, 5).astype(bool)
    img[func, :] = 0
    # io.imshow(img)
    #     # plt.show()

    contours = measure.find_contours(rgb2gray(img), 0.2)
    new_contours = check_if_circle(contours, 100, 200, 5000)
    new_contours = find_n_greatest_areas(new_contours, n_times=4)
    # draw_contours(img_source, new_contours, 'navy')
    return new_contours


# -------START

names = ['ex1', 'ex2', 'ex3', 'ex4', 'ex5', 'ex6', 'ex71', 'ex72']
medium = ['ex1', 'ex3', 'ex4', 'ex72']
hard = ['ex3', 'ex4', 'ex72']
blue = ['ex3', 'ex71', 'ex72']
newlist = ['ex81']

for name in newlist:
    img_name = name + '.jpg'
    img = io.imread(os.path.join('resources/', img_name))
    img_board = BoardAssignment(img)
    draw_counters_and_circles(img_board, img, red=True, white=False, black=True, circle=True, orange=True, fields=True)
    #detect_grain_field(img_board)

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
