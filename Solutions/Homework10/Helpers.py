import numpy as np


def get_integral_image(img):
    row_sum = np.zeros(img.shape)
    integral_image = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.int64)

    for y, row in enumerate(img):
        for x, col in enumerate(row):
            row_sum[y, x] = row_sum[y - 1, x] + img[y, x]
            integral_image[y + 1, x + 1] = integral_image[y + 1, x - 1 + 1] + row_sum[y, x]

    return integral_image


def get_integral_img_sub_sum(integral_image, top_left_position, bottom_right_position):
    # as x and y coordinates in an image are flipped, therefore they are flipped within the integral image as well
    # as you can see in the upper method, we take the rows as y and cols as x, but in the final result we leave
    # them as they are

    top_left = top_left_position[1], top_left_position[0]
    bottom_right = bottom_right_position[1], bottom_right_position[0]

    # sum of a single cell, as coords of top left and bottom right corner of sub-image are identical
    if top_left == bottom_right:
        return integral_image[top_left]

    top_right = bottom_right[0], top_left[1]
    bottom_left = top_left[0], bottom_right[1]

    return np.int64(np.int64(integral_image[bottom_right]) - np.int64(integral_image[top_right]) -
                    np.int64(integral_image[bottom_left]) + np.int64(integral_image[top_left]))
