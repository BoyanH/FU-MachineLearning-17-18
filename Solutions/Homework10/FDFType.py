from enum import Enum


class FDFType(Enum):
    TWO_RECTANGLE_HORIZONTAL = 1
    TWO_RECTANGLE_VERTICAL = 2
    THREE_RECTANGLE_HORIZONTAL = 3
    THREE_RECTANGLE_VERTICAL = 4

    # it makes no sense to differ between horizontal and vertical here
    # as it is the difference between the diagonals
    # positive or negative, it is the same number, we don't really care
    FOUR_RECTANGLE = 5
