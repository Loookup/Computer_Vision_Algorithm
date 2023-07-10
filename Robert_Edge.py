import copy
import cv2
import numpy as np

def Robert_Edge_Detect(img):

    Result_x, Result_y, Result = copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img)

    mask_x, mask_y = np.zeros((3, 3)).astype('float32'), np.zeros((3, 3)).astype('float32')

    mask_x[0][0], mask_x[1][1] = -1, 1

    mask_y[0][2], mask_y[1][1] = -1, 1

    print(mask_x)

    print(mask_y)

    rows, cols = img.shape[:2]

    for i in range(2, rows-1):

        for j in range(2, cols-1):

            temp = img[i-1:i+2, j-1:j+2].astype('float32')

            Result_x[i][j] = np.sum(cv2.multiply(mask_x, temp))

            Result_y[i][j] = np.sum(cv2.multiply(mask_y, temp))

            value = abs(np.sum(cv2.multiply(mask_x, temp))) + abs(np.sum(cv2.multiply(mask_y, temp)))

            Result[i][j] = value

    return Result_x, Result_y, Result


img = cv2.imread('Landscape.jpeg')

if img is None:
    raise Exception("Read Error")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Origin', gray)

Result_x, Result_y, Result = Robert_Edge_Detect(gray)

cv2.imshow('Robert_x', Result_x)

cv2.imshow('Robert_y', Result_y)

cv2.imshow('Robert', Result)

cv2.waitKey(0)

cv2.destroyAllWindows()