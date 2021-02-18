
import math


class Evaluator(object):

    @staticmethod
    def rectangle_evaluation(x: int, y: int, p1: tuple, p2: tuple, **kwargs) -> bool:
        (x_min, y_min), (x_max, y_max) = p1, p2
        if x_min <= x < x_max and y_min <= y < y_max:
            return True

    @staticmethod
    def triangle_evaluation(x: int, y: int, p1: tuple, p2: tuple, p3: tuple, **kwargs) -> bool:
        full = abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        first = abs(p1[0] * (p2[1] - y) + p2[0] * (y - p1[1]) + x * (p1[1] - p2[1]))
        second = abs(p1[0] * (y - p3[1]) + x * (p3[1] - p1[1]) + p3[0] * (p1[1] - y))
        third = abs(x * (p2[1] - p3[1]) + p2[0] * (p3[1] - y) + p3[0] * (y - p2[1]))
        return abs(first + second + third - full) < .0000000001

    @staticmethod
    def empty_circle_evaluation(x: int, y: int, p1: tuple, radius: float, line_width: float, **kwargs):
        (center_x, center_y) = p1

        if abs((x - center_x)**2 + (y - center_y)**2 - radius**2) < line_width:
            return True

    @staticmethod
    def filled_circle_evaluation(x: int, y: int, p1: tuple, radius: float, ** kwargs):
        (center_x, center_y) = p1

        if (x - center_x)**2 + (y - center_y)**2 < radius**2:
            return True
