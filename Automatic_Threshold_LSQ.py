import copy
import cv2
import cv2.aruco as aruco
import numpy as np

aruco_id = 0
marker_size = 0.1  # [m]

mtx = np.array([[644.04776251, 0, 314.22661835], [0, 640.76025288, 229.87075642], [0, 0, 1]])
dist = np.array([[0.02758655, -0.09118903, 0.00130404, -0.00147594, -0.57083165]]) # k1, k2, p1, p2


def onChange_LSQ(value):

    global img, title, Width, Height, Result, origin, Gain_List

    gain = value / 100

    img_copy = copy.deepcopy(img)

    img_copy = Tunning(Width, Height, Result, img_copy, gain)

    corners, ids, rejected = aruco.detectMarkers(image=img_copy, dictionary=aruco_dict,
                                                     parameters=parameters, cameraMatrix=mtx, distCoeff=dist)

    if ids != None and ids[0] == aruco_id:

        origin_copy = copy.deepcopy(origin)

        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)

        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

        aruco.drawDetectedMarkers(origin_copy, corners)

        if gain in Gain_List:
            print('')
        else:
            Gain_List.append(gain)


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


def onChange_Thres(value):

    global img, title, Width, Height, origin, Threshold_List

    Threshold = value

    img_copy = copy.deepcopy(img)

    img_copy = Tunning_Thres(Width, Height, img_copy, Threshold)

    corners, ids, rejected = aruco.detectMarkers(image=img_copy, dictionary=aruco_dict,
                                                     parameters=parameters, cameraMatrix=mtx, distCoeff=dist)

    if ids != None and ids[0] == aruco_id:

        # origin_copy = copy.deepcopy(origin)
        #
        # ret = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)
        #
        # rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]
        #
        # aruco.drawDetectedMarkers(origin_copy, corners)

        if Threshold in Threshold_List:
            print('')
        else:
            Threshold_List.append(Threshold)


def Tunning_Thres(Width, Height, img, threshold):

    idx = 0

    for W_idx in range(Width):

        for H_idx in range(Height):

            if(img[H_idx][W_idx] >= threshold):
                img[H_idx][W_idx] = 255

            else:
                img[H_idx][W_idx] = 0
            idx += 1

    return img


title = 'LSQ'

bar_name = 'Gain'

Gain_List = []

Threshold_List = []

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

parameters = aruco.DetectorParameters_create()

origin = cv2.imread('ShadowedAruco.jpeg')

img = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

Width, Height, Result = Linear_LSQ(img)

for idx in range(100):
    onChange_LSQ(idx)

for idx in range(255):
    onChange_Thres(idx)

print('Gain List : ')

print(Gain_List)

print('Threshold List : ')

print(Threshold_List)
