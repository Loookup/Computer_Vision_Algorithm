import cv2
import numpy as np
import copy
import os

sat = cv2.imread("dock5.jpg")

sat1 = copy.deepcopy(sat)

gray1 = cv2.cvtColor(sat1, cv2.COLOR_BGR2GRAY)

gray1 = np.float32(gray1)

dst = cv2.cornerHarris(gray1, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

sat1[dst > 0.01*dst.max()] = [0, 0, 255]

cv2.imshow("Win1", sat1)

# ret, th_sat1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
# 
# cv2.imshow("Win2", th_sat1)
# 

sat2 = copy.deepcopy(sat)

gray2 = cv2.cvtColor(sat1, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray2, 250, 0.01, 10)
corners = np.intp(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(sat2, (x, y), 3, 255, -1)

cv2.imshow("Win2", sat2)

cv2.namedWindow("Win3")


sat3 = copy.deepcopy(sat)

gray3 = cv2.cvtColor(sat3, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(400)

keypoints, desc = surf.detectAndCompute(gray3, None)

print(len(keypoints))

img_draw = cv2.drawKeypoints(sat3, keypoints, None, \
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SURF', img_draw)

# def nothing():
#     pass

# cv2.createTrackbar('low threshold', 'Win3', 0, 1000, nothing)

# cv2.createTrackbar('high threshold', 'Win3', 0, 1000, nothing)

# cv2.setTrackbarPos('low threshold', 'Win3', 50)

# cv2.setTrackbarPos('high threshold', 'Win3', 150)

# sat3 = copy.deepcopy(sat)

# gray3 = cv2.cvtColor(sat1, cv2.COLOR_BGR2GRAY)

# while True:

#     low = cv2.getTrackbarPos('low threshold', 'Win3')
#     high = cv2.getTrackbarPos('high threshold', 'Win3')

#     edges = cv2.Canny(gray3, low, high)

#     # cv2.imshow("Win3", edges)


key = cv2.waitKey(0)

if key == 27:
    cv2.destroyAllWindows()
    os._exit(1)

        # cv2.imwrite("Harrris.jpg", sat1)

        # cv2.imwrite("Shi&Tomasi.jpg", sat2)
