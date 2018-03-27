from shapely.geometry import Polygon
from utils import *

def PIOU(a, b):
    def change_to_corners(c_x, c_y, w, h, angle):
        w_c = w * np.cos(angle)
        w_s = w * np.sin(angle)
        h_c = h * np.cos(angle)
        h_s = h * np.sin(angle)

        x1 = c_x + (-w_c - h_s) / 2
        x2 = c_x + (-w_c + h_s) / 2
        x3 = c_x + (w_c + h_s) / 2
        x4 = c_x + (w_c - h_s) / 2

        y1 = c_y + (-w_s + h_c) / 2
        y2 = c_y + (-w_s - h_c) / 2
        y3 = c_y + (w_s - h_c) / 2
        y4 = c_y + (w_s + h_c) / 2

        return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    a_arr = change_to_corners(a[0], a[1], a[2], a[3], a[4])
    b_arr = change_to_corners(b[0], b[1], b[2], b[3], b[4])
    P_a = Polygon(a_arr)
    P_b = Polygon(b_arr)
    inter = P_a.intersection(P_b).area
    print(inter)
    union = P_a.area + P_b.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

a = [0, 0, 20, 10, 0]
b = [0, 0, 20, 10, 3.14/2]
print(PIOU(a, b))