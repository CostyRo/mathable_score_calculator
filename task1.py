from sys import argv
from os.path import join as path_join

import cv2
import numpy as np

img = cv2.imread(path_join("antrenare","1_01.jpg"),cv2.IMREAD_GRAYSCALE)
blured_img = cv2.GaussianBlur(img,(3,3),0)
canny = cv2.Canny(blured_img,50,150)
lines = cv2.HoughLinesP(
    canny,
    rho=1,
    theta=np.pi/180,
    threshold=10,
    minLineLength=20,
    maxLineGap=10
)

min_x1,_,min_x2,_ = min(lines,key=lambda x: min(x[0][0],x[0][2]))[0]
max_x1,_,max_x2,_ = max(lines,key=lambda x: max(x[0][0],x[0][2]))[0]
min_x = min(min_x1,min_x2)
max_x = max(max_x1,max_x2)

_,min_y1,_,min_y2 = min(lines,key=lambda x: min(x[0][1],x[0][3]))[0]
_,max_y1,_,max_y2 = max(lines,key=lambda x: max(x[0][1],x[0][3]))[0]
min_y = min(min_y1,min_y2)
max_y = max(max_y1,max_y2)

if len(argv) >= 2 and argv[1] == "debug":
  for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 20)
  cv2.namedWindow("image",cv2.WINDOW_NORMAL)
  cv2.imshow("image",img[min_y:max_y,min_x:max_x])
  cv2.waitKey(0)
  cv2.destroyAllWindows()
