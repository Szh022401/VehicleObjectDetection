def get_center_of_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_box_width(box):
    x1, _, x2, _ = box
    return abs(x2 - x1)




def measure_distance(box1, box2):
    return ((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2) ** 0.5


def get_car_position(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]