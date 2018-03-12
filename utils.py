# coding:utf-8
import tensorflow as tf
import numpy as np
import cv2

# def batch_norm(x, is_training, decay=0.9, eps=1e-5):
#     with tf.variable_scope("batch_norm"):
#         shape = x.get_shape().as_list()
#         assert len(shape) in [2, 3, 4]
#         norm_num = shape[-1]
#         beta = tf.get_variable(name="beta", shape=[norm_num],
#                                dtype=tf.float32, initializer=tf.constant_initializer(0))
#         gamma = tf.get_variable(name="beta", shape=[norm_num],
#                                 dtype=tf.float32, initializer=tf.constant_initializer(1))
#         if len(shape) == 2:
#             batch_mean, batch_var = tf.nn.moments(x, [0])
#         elif len(shape) == 3:
#             batch_mean, batch_var = tf.nn.moments(x, [0, 1])
#         else:
#             batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
#         ema = tf.train.ExponentialMovingAverage(decay=decay)
#
#         def mean_var_update_when_training():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)
#
#         def mean_var_update_when_not_training():
#             return ema.average(batch_mean), ema.average(batch_var)
#
#         mean, var = tf.cond(is_training, mean_var_update_when_training, mean_var_update_when_not_training)
#
#         return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)


def get_out1_shape(img_height, img_width):
    out_h = int((int((int((img_height + 0.5)/2) + 0.5)/2) + 0.5)/2)
    out_w = int((int((int((img_width + 0.5)/2) + 0.5)/2) + 0.5)/2)
    return out_h, out_w


def get_out2_shape(img_height, img_width):
    out_h, out_w = get_out1_shape(img_height, img_width)
    out_h = int((out_h + 0.5)/2)
    out_w = int((out_w + 0.5)/2)
    return out_h, out_w


def get_out3_shape(img_height, img_width):
    out_h, out_w = get_out2_shape(img_height, img_width)
    out_h = int((out_h + 0.5) / 2)
    out_w = int((out_w + 0.5) / 2)
    return out_h, out_w


def get_out4_shape(img_height, img_width):
    out_h, out_w = get_out3_shape(img_height, img_width)
    out_h = int((out_h + 0.5) / 2)
    out_w = int((out_w + 0.5) / 2)
    return out_h, out_w


def get_out5_shape(img_height, img_width):
    out_h, out_w = get_out4_shape(img_height, img_width)
    out_h = int((out_h + 0.5) / 2)
    out_w = int((out_w + 0.5) / 2)
    return out_h, out_w


def get_out6_shape(img_height, img_width):
    out_h, out_w = get_out5_shape(img_height, img_width)
    return out_h, out_w


def mrec_centre_To_mrec_corners(c_x, c_y, w, h, angle):

    w_c = w * np.cos(angle)
    w_s = w * np.sin(angle)
    h_c = h * np.cos(angle)
    h_s = h * np.sin(angle)

    x1 = int(c_x + (-w_c - h_s) / 2)
    x2 = int(c_x + (-w_c + h_s) / 2)
    x3 = int(c_x + (w_c + h_s) / 2)
    x4 = int(c_x + (w_c - h_s) / 2)

    y1 = int(c_y + (-w_s + h_c) / 2)
    y2 = int(c_y + (-w_s - h_c) / 2)
    y3 = int(c_y + (w_s - h_c) / 2)
    y4 = int(c_y + (w_s + h_c) / 2)

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]


def mrec_centre_To_mrec_corners_L(data_list):

    c_x = data_list[0]
    c_y = data_list[1]
    w = data_list[2]
    h = data_list[3]
    angle = data_list[4]
    return mrec_centre_To_mrec_corners(c_x, c_y, w, h, angle)


def draw_min_rect(img, anns):
    tst_contours = []
    print(len(anns))
    for ann in anns:
        print(ann[1])
        tmp_contour = mrec_centre_To_mrec_corners_L(ann[1])
        tmp_contour = np.asarray(tmp_contour)
        tst_contours.append(tmp_contour)
    res_img = cv2.drawContours(img, tst_contours, -1, (255, 0, 0), 2)
    return res_img

def mrec_centre_To_rec_centre(c_x, c_y, w, h, angle):
    w_res = np.abs(h*np.sin(angle)) + w*np.cos(angle)
    h_res = np.abs(w*np.sin(angle)) + h*np.cos(angle)

    w_res = int(w_res)
    h_res = int(h_res)

    return [int(c_x), int(c_y), w_res, h_res]

def mrec_centre_To_rec_centre_L(data_list):
    c_x = data_list[0]
    c_y = data_list[1]
    w = data_list[2]
    h = data_list[3]
    angle = data_list[4]
    return mrec_centre_To_rec_centre(c_x, c_y, w, h, angle)


def mrec_centre_To_rec_corner(c_x, c_y, w, h, angle):
    w_res = np.abs(h*np.sin(angle)) + w*np.cos(angle)
    h_res = np.abs(w*np.sin(angle)) + h*np.cos(angle)
    x_res = c_x - w_res/2
    y_res = c_y - h_res/2

    w_res = int(w_res)
    h_res = int(h_res)
    x_res = int(x_res)
    y_res = int(y_res)

    return [x_res, y_res, w_res, h_res]

def mrec_centre_To_rec_corner_L(data_list):
    c_x = data_list[0]
    c_y = data_list[1]
    w = data_list[2]
    h = data_list[3]
    angle = data_list[4]
    return mrec_centre_To_rec_corner(c_x, c_y, w, h, angle)

def rec_corner_To_corners(x, y, w, h):
    x1 = int(x)
    x2 = int(x + w)
    x3 = int(x + w)
    x4 = int(x)

    y1 = int(y)
    y2 = int(y)
    y3 = int(y + h)
    y4 = int(y + h)

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

def rec_corner_To_corners_L(data_list):
    x = data_list[0]
    y = data_list[1]
    w = data_list[2]
    h = data_list[3]

    return rec_corner_To_corners(x, y, w, h)

def rec_centre_To_corners(x, y, w, h):
    x1 = int(x - w/2)
    x2 = int(x + w/2)
    x3 = int(x + w/2)
    x4 = int(x - w/2)

    y1 = int(y - h/2)
    y2 = int(y - h/2)
    y3 = int(y + h/2)
    y4 = int(y + h/2)

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]


def rec_centre_To_corners_L(data_list):
    x = data_list[0]
    y = data_list[1]
    w = data_list[2]
    h = data_list[3]
    return rec_centre_To_corners(x, y, w, h)

def rec_centre_To_rec_corner(x, y, w, h):
    res_x = x - w/2
    res_y = y - h/2
    return [res_x, res_y, w, h]

def rec_centre_To_rec_corner_L(data_list):
    x = data_list[0]
    y = data_list[1]
    w = data_list[2]
    h = data_list[3]
    return rec_centre_To_rec_corner(x, y, w, h)

def clip_box(r):
    return [r[0], r[1], max(r[2], 0.01), max(r[3], 0.01)]

def calc_jaccard(r1, r2):# 这里的需要传左上点
    r1_ = clip_box(r1)
    r2_ = clip_box(r2)
    intersection = calc_intersection(r1_, r2_)
    union = r1_[2] * r1_[3] + r2_[2] * r2_[3] - intersection

    if union <= 0:
        return 0

    j = intersection / union

    return j

def calc_intersection(r1, r2):
    left = max(r1[0], r2[0])
    right = min(r1[0] + r1[2], r2[0] + r2[2])
    bottom = min(r1[1] + r1[3], r2[1] + r2[3])
    top = max(r1[1], r2[1])

    if left < right and top < bottom:
        return (right - left) * (bottom - top)

    return 0

if __name__ == '__main__':
    print(get_out6_shape(256, 256))