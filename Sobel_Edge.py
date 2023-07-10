import copy
import cv2
import numpy as np
import cv2.aruco as aruco

aruco_id = 0
marker_size = 0.1  # [m]

mtx = np.array([[644.04776251, 0, 314.22661835], [0, 640.76025288, 229.87075642], [0, 0, 1]])
dist = np.array([[0.02758655, -0.09118903, 0.00130404, -0.00147594, -0.57083165]]) # k1, k2, p1, p2

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
parameters = aruco.DetectorParameters_create()

def Sobel_Edge_Detect(img):

    Result_x, Result_y, Result = copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img)

    mask_x, mask_y = np.zeros((3, 3)).astype('float32'), np.zeros((3, 3)).astype('float32')

    mask_x[0:3, 0], mask_x[0:3, 2] = -1, 1

    mask_x[1, 0:3] = mask_x[1, 0:3]*2

    mask_y[0, 0:3], mask_y[2, 0:3] = -1, 1

    mask_y[0:3, 1] = mask_y[0:3, 1] * 2

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

    return Result


img = cv2.imread('blight2.jpeg')

if img is None:
    raise Exception("Read Error")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Origin', gray)

Result = Sobel_Edge_Detect(gray)

# cv2.imshow('X_Direction', Result_x)
#
# cv2.imshow('Y_Direction', Result_y)

cv2.imshow('Full', Result)

corners, ids, rejected = aruco.detectMarkers(image=Result, dictionary=aruco_dict,
                                                 parameters=parameters, cameraMatrix=mtx, distCoeff=dist)

if ids != None and ids[0] == aruco_id:

    ret = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)

    rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

    aruco.drawDetectedMarkers(img, corners)

    aruco.drawAxis(img, mtx, dist, rvec, tvec, 0.07)

else:

    print('Failed')

cv2.waitKey(0)

cv2.destroyAllWindows()
