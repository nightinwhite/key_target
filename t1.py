import numpy as np
import os
tmp_path = "/home/hp/Data/train_data/train_box_anns_new"
tmp_names = os.listdir(tmp_path)
for n in tmp_names:
    tmp_arr = np.load(tmp_path + "/" + n)
    print(n)
    for i in tmp_arr:
        if (i != [0, 0, 0, 0]).all():
            print(i)
