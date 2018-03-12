import os
import numpy as np
path = "D:/data/train_data/slice_class_anns/"
file_names = os.listdir(path)
data_dict = {}
for f_name in file_names:
    tmp_f = np.load(path + f_name)
    for d in tmp_f:
        tmp_num = data_dict.get(d,0)
        tmp_num += 1
        data_dict[d] = tmp_num
print(data_dict)