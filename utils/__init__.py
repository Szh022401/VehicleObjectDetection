from .box_utils import *
from .video_utils import *


def get_center_of_box(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def get_box_width(box):
    x1, _, x2, _ = box
    return abs(x2 - x1)


def get_foot_position(box):
    return None
