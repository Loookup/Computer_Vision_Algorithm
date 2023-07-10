import cv2
import numpy as np
import math

mtx = np.array([[644.04776251, 0, 314.22661835], [0, 640.76025288, 229.87075642], [0, 0, 1]])

dist = np.array([[0.02758655, -0.09118903, 0.00130404, -0.00147594, -0.57083165]])


def Undistort_Image(img, mtx, dist):

    h, w = 480, 640        # h : height of Image, w : width of Image

    img_correct = np.ones((h, w), np.uint8)

    fx, fy, cx, cy = mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]

    inv_mtx = np.linalg.inv(mtx)

    inv_fx, inv_fy, inv_cx, inv_cy = inv_mtx[0, 0], inv_mtx[1, 1], inv_mtx[0, 2], inv_mtx[1, 2]

    k1, k2, p1, p2, k3 = dist[0, 0], dist[0, 1], dist[0, 2], dist[0, 3], dist[0, 4]

    for y_img_undist in range(0, h):

        y_nor_undist = inv_fy * y_img_undist + inv_cy

        for x_img_undist in range(0, w):

            x_nor_undist = inv_fx * x_img_undist + inv_cx

            rad_undist_2 = x_nor_undist*x_nor_undist + y_nor_undist*y_nor_undist

            radial_dist = 1 + k1*rad_undist_2 + k2*rad_undist_2*rad_undist_2 + k3*rad_undist_2*rad_undist_2*rad_undist_2

            x_nor_dist = radial_dist*x_nor_undist + 2*p1*x_nor_undist*y_nor_undist +\
                         p2*(rad_undist_2 + 2*x_nor_undist*x_nor_undist)

            y_nor_dist = radial_dist*y_nor_undist + 2*p2*x_nor_undist*y_nor_undist +\
                         p1*(rad_undist_2 + 2*y_nor_undist*y_nor_undist)

            x_img_dist = fx*x_nor_dist + cx

            y_img_dist = fy*y_nor_dist + cy

            idx_x_img_dist, idx_y_img_dist = round(x_img_dist), round(y_img_dist)

            img_correct[y_img_undist, x_img_undist] = img[idx_y_img_dist, idx_x_img_dist]

    print("*")

    return img_correct


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:

    ret, frame = capture.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_correct = Undistort_Image(frame, mtx, dist)

    cv2.imshow("Video", frame_correct)

    if cv2.waitKey(10) == 27:
        break
    # elif cv2.waitKey(10) == ord('c'):
    #     num += 1
    #     print("%s Cap"%num)
    #     img_cap = cv2.imwrite('cap' + str(num) + '.jpg', frame)

capture.release()

cv2.destroyAllWindows()