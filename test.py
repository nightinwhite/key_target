# coding:utf-8
import cv2
all_mn_path = "D:\\data\\key_target_data\\micai\\guonei\\mn1.tif"
img = cv2.imread(all_mn_path)
cv2.imshow("0", img)
cv2.waitKey(0)
cv2.destroyAllWindows()