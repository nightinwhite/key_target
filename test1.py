import cv2
from parse_anns import parse_xml
from utils import *

img = cv2.imread("tst_data/mw_227.tif")
anns = parse_xml("tst_data/mw_227.xml")
tst_contours = []
print(len(anns))
for ann in anns:
    print(ann[1])
    tmp_contour = mrec_centre_To_mrec_corners_L(ann[1])
    tmp_contour = np.asarray(tmp_contour)
    tst_contours.append(tmp_contour)
img = cv2.drawContours(img, tst_contours, -1, (255, 0, 0), 2)
img = cv2.resize(img, (600, 600))
cv2.imshow("1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()