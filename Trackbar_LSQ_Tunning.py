import copy
import cv2
import cv2.aruco as aruco
import numpy as np

aruco_id = 0
marker_size = 0.1  # [m]

mtx = np.array([[644.04776251, 0, 314.22661835], [0, 640.76025288, 229.87075642], [0, 0, 1]])
dist = np.array([[0.02758655, -0.09118903, 0.00130404, -0.00147594, -0.57083165]]) # k1, k2, p1, p2


def ArucoDetect(img):

    corners, ids, rejected = aruco.detectMarkers(image=img, dictionary=aruco_dict,
                                                 parameters=parameters, cameraMatrix=mtx, distCoeff=dist)

    if ids != None and ids[0] == aruco_id:

        origin_copy = copy.deepcopy(origin)

        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)

        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

        aruco.drawDetectedMarkers(origin_copy, corners)

        aruco.drawAxis(origin_copy, mtx, dist, rvec, tvec, 0.07)

    else:

        print('Failed')


def onChange(value):

    global img, title, Width, Height, Result, origin, Gain_List

    gain = value / 100

    img_copy = copy.deepcopy(img)

    img_copy = Tunning(Width, Height, Result, img_copy, gain)

    cv2.imshow(title, img_copy)

    corners, ids, rejected = aruco.detectMarkers(image=img_copy, dictionary=aruco_dict,
                                                     parameters=parameters, cameraMatrix=mtx, distCoeff=dist)

    if ids != None and ids[0] == aruco_id:

        origin_copy = copy.deepcopy(origin)

        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)

        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

        aruco.drawDetectedMarkers(origin_copy, corners)

        aruco.drawAxis(origin_copy, mtx, dist, rvec, tvec, 0.07)

        cv2.imshow('Result', origin_copy)

        if gain not in Gain_List:
            Gain_List.append(gain)

    else:

        cv2.imshow('Result', origin)


def Linear_LSQ(img):

    Height, Width = img.shape[0], img.shape[1]

    Pixels = np.zeros((Height * Width, 1), dtype=np.uint8)

    Location = np.ones((Height * Width, 3), dtype=np.uint8)

    idx = 0

    for W_idx in range(Width):

        for H_idx in range(Height):
            Location[idx][0], Location[idx][1] = W_idx, H_idx
            Pixels[idx] = img[H_idx][W_idx]

            idx += 1

    pinvA = np.linalg.pinv(Location)

    X = np.dot(pinvA, Pixels)

    Result = np.dot(Location, X)

    return Width, Height, Result


def Tunning(Width, Height, Result, img, gain):

    idx = 0

    for W_idx in range(Width):

        for H_idx in range(Height):

            if(img[H_idx][W_idx] >= Result[idx]*gain):
                img[H_idx][W_idx] = 255

            else:
                img[H_idx][W_idx] = 0
            idx += 1

    return img


title = 'LSQ'

bar_name = 'Gain'

Gain_List = []

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

parameters = aruco.DetectorParameters_create()

origin = cv2.imread('ShadowedAruco.jpeg')

cv2.imshow('Result', origin)

img = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

ArucoDetect(img)

cv2.imshow(title, img)

Width, Height, Result = Linear_LSQ(img)

cv2.createTrackbar('Gain', title, 100, 100, onChange)

cv2.waitKey(0)

cv2.destroyAllWindows()

print(Gain_List)
