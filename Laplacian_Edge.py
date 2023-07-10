import copy
import cv2
import numpy as np


def Laplacian_Edge_Detect(img):

    Result_4, Result_8 = copy.deepcopy(img), copy.deepcopy(img)

    mask_4, mask_8 = np.zeros((3, 3)).astype('float32'), np.ones((3, 3)).astype('float32')

    mask_4[1, 0:3], mask_4[0:3, 1], mask_4[1][1] = 1, 1, -4

    mask_8[1][1] = -8

    print(mask_4)

    print(mask_8)

    rows, cols = img.shape[:2]

    for i in range(2, rows-1):

        for j in range(2, cols-1):

            temp = img[i-1:i+2, j-1:j+2].astype('float32')

            Result_4[i][j] = abs(np.sum(cv2.multiply(mask_4, temp)))

            Result_8[i][j] = abs(np.sum(cv2.multiply(mask_8, temp)))

            # value = abs(np.sum(cv2.multiply(mask_4_x, temp))) + abs(np.sum(cv2.multiply(mask_4_y, temp)))

    return Result_4, Result_8


img = cv2.imread('Landscape.jpeg')

if img is None:
    raise Exception("Read Error")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Origin', gray)

Result_4, Result_8 = Laplacian_Edge_Detect(gray)

# cv2.imshow('X_Direction', Result_x)
#
# cv2.imshow('Y_Direction', Result_y)

cv2.imshow('Full', Result_4)

cv2.imshow('Full2', Result_8)

cv2.waitKey(0)

cv2.destroyAllWindows()
