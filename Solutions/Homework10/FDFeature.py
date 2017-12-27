from FDFType import FDFType
from Helpers import get_integral_img_sub_sum


class FDFeature:
    def __init__(self, width_percent, height_percent, type, pos_x_percent, pos_y_percent):
        self.wp = width_percent
        self.hp = height_percent
        self.type = type
        self.xp = pos_x_percent
        self.yp = pos_y_percent

    def get_value(self, integral_image):
        img_width = integral_image.shape[1] - 1
        img_height = integral_image.shape[0] - 1

        pos_x = int(self.xp * img_width)
        pos_y = int(self.yp * img_height)
        top_left = pos_x, pos_y

        br_x = int(pos_x + self.wp * img_width)
        br_y = int(pos_y + self.hp * img_height)
        bottom_right = br_x, br_y

        if self.type == FDFType.TWO_RECTANGLE_HORIZONTAL:
            middle_left = top_left[0], int(top_left[1] + self.hp * img_width / 2)
            middle_right = bottom_right[0], middle_left[1]
            a = get_integral_img_sub_sum(integral_image, top_left, middle_right)
            b = get_integral_img_sub_sum(integral_image, middle_left, bottom_right)

            return a - b
        elif self.type == FDFType.TWO_RECTANGLE_VERTICAL:
            middle_top = int(top_left[0] + img_width / 2 * self.wp), top_left[1]
            middle_bottom = middle_top[0], bottom_right[1]

            a = get_integral_img_sub_sum(integral_image, top_left, middle_bottom)
            b = get_integral_img_sub_sum(integral_image, middle_top, bottom_right)

            return a - b
        elif self.type == FDFType.THREE_RECTANGLE_HORIZONTAL:
            one_third_left = top_left[0], int(top_left[1] + 1 / 3 * img_height * self.hp)
            one_third_right = bottom_right[0], one_third_left[1]
            two_thirds_left = top_left[0], int(top_left[1] + 2 / 3 * img_height * self.hp)
            two_thirds_right = bottom_right[0], two_thirds_left[1]

            a = get_integral_img_sub_sum(integral_image, top_left, one_third_right)
            b = get_integral_img_sub_sum(integral_image, one_third_left, two_thirds_right)
            c = get_integral_img_sub_sum(integral_image, two_thirds_left, bottom_right)

            return a - b + c
        elif self.type == FDFType.THREE_RECTANGLE_VERTICAL:
            one_third_top = int(top_left[0] + 1 / 3 * self.wp * img_width), top_left[1]
            one_third_bottom = one_third_top[0], bottom_right[1]
            two_thirds_top = int(top_left[0] + 2 / 3 * self.wp * img_width), top_left[1]
            two_thirds_bottom = two_thirds_top[0], bottom_right[1]

            a = get_integral_img_sub_sum(integral_image, top_left, one_third_bottom)
            b = get_integral_img_sub_sum(integral_image, one_third_top, two_thirds_bottom)
            c = get_integral_img_sub_sum(integral_image, two_thirds_top, bottom_right)

            return a - b + c
        elif self.type == FDFType.FOUR_RECTANGLE:
            middle_left = top_left[0], int(top_left[1] + self.hp * img_width / 2)
            middle_right = bottom_right[0], middle_left[1]
            middle_top = int(top_left[0] + img_width / 2 * self.wp), top_left[1]
            middle_bottom = middle_top[0], bottom_right[1]
            middle = middle_top[0], middle_left[1]

            a = get_integral_img_sub_sum(integral_image, top_left, middle)
            b = get_integral_img_sub_sum(integral_image, middle_top, middle_right)
            c = get_integral_img_sub_sum(integral_image, middle_left, middle_bottom)
            d = get_integral_img_sub_sum(integral_image, middle, bottom_right)

            return (a + d) - (b + c)

        return 0
