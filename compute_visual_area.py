def single_compute(ori_visual_area, ori_stride, watch_area, watch_stride):
    res_visual_x = (watch_area[0] - 1) * ori_stride[0] + 1 + ori_visual_area[0]//2*2
    res_visual_y = (watch_area[1] - 1) * ori_stride[1] + 1 + ori_visual_area[1]//2*2
    res_stride_x = ori_stride[0]*watch_stride[0]
    res_stride_y = ori_stride[1]*watch_stride[1]
    return [res_visual_x, res_visual_y],[res_stride_x,res_stride_y]


def VGG_16_visual_area():
    tmp_visual_area = [1, 1]
    tmp_stride = [1, 1]
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [2, 2], [2, 2])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [2, 2], [2, 2])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [2, 2], [2, 2])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [2, 2], [2, 2])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    print(tmp_visual_area, tmp_stride)

def inception_v3_min():
    tmp_visual_area = [1, 1]
    tmp_stride = [1, 1]
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    # p8_a
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    # p8_b
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    # p8_c
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    # p9_a
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    # p9_b
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    # p9_c
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    # p9_d
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    # p9_e
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    # p10_a
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    # p10_b
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    # p10_c
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    print(tmp_visual_area, tmp_stride)

def inception_v3_max():
    tmp_visual_area = [1, 1]
    tmp_stride = [1, 1]
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    print(tmp_visual_area, tmp_stride)
    # p8_a
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [5, 5], [1, 1])
    print(tmp_visual_area, tmp_stride)
    # p8_b
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [5, 5], [1, 1])
    print(tmp_visual_area, tmp_stride)
    # p8_c
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [5, 5], [1, 1])
    # p9_a
    print(tmp_visual_area, tmp_stride)
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [5, 5], [2, 2])
    # p9_b
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [7, 7], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [7, 7], [1, 1])
    # p9_c
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [7, 7], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [7, 7], [1, 1])
    # p9_d
    print(tmp_visual_area, tmp_stride)
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [7, 7], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [7, 7], [1, 1])
    # p9_e
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [7, 7], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [7, 7], [1, 1])
    # p10_a
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [7, 7], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    print(tmp_visual_area, tmp_stride)
    # p10_b
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    # p10_c
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    print(tmp_visual_area, tmp_stride)

def inception_new_min():
    tmp_visual_area = [1, 1]
    tmp_stride = [1, 1]
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    print(tmp_visual_area, tmp_stride)
    # p8_a
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    print(tmp_visual_area, tmp_stride)
    # p8_b
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    print(tmp_visual_area, tmp_stride)
    # p9_a
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    print(tmp_visual_area, tmp_stride)
    # p9_b
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    print(tmp_visual_area, tmp_stride)
    # p10_a
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    print(tmp_visual_area, tmp_stride)

def inception_new_max():
    tmp_visual_area = [1, 1]
    tmp_stride = [1, 1]
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [1, 1], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    print(tmp_visual_area, tmp_stride)
    # p8_a
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [5, 5], [2, 2])
    print(tmp_visual_area, tmp_stride)
    # p8_b
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [5, 5], [2, 2])
    print(tmp_visual_area, tmp_stride)
    # p9_a
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [5, 5], [2, 2])
    print(tmp_visual_area, tmp_stride)
    # p9_b
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [7, 7], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [7, 7], [2, 2])
    print(tmp_visual_area, tmp_stride)
    # p10_a
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [7, 7], [1, 1])
    tmp_visual_area, tmp_stride = single_compute(tmp_visual_area, tmp_stride, [3, 3], [2, 2])
    print(tmp_visual_area, tmp_stride)
if __name__ == '__main__':
    inception_new_min()
    print("_________________________________")
    inception_new_max()


