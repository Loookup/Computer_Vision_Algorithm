import math

import cv2
import numpy as np


# Intrinsic Parameter, (640 * 480)
mtx = np.array([[644.04776251, 0, 314.22661835], [0, 640.76025288, 229.87075642], [0, 0, 1]])

dist = np.array([[0.02758655, -0.09118903, 0.00130404, -0.00147594, -0.57083165]]) # k1, k2, p1, p2, k3

points_2D = np.array([(179, 273), (415, 314), (260, 100), (467, 109)], dtype="double")

points_3D = np.array([(0, 0, 0), (5, 0, 0), (0, 5, 0), (5, 5, 0)], dtype="double")

retval, rvec, tvec = cv2.solvePnP(points_3D, points_2D, mtx, dist, None, None, None, None)

print(rvec)

or_rvec = rvec

norm = np.linalg.norm(rvec)

rvec = rvec / norm

id_mat = np.zeros(shape=(3, 3))

np.fill_diagonal(id_mat, val=1)

rx, ry, rz = rvec[0][0], rvec[1][0], rvec[2][0]

z = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])

my_R = math.cos(norm)*id_mat + (1 - math.cos(norm))*rvec*np.transpose(rvec) + math.sin(norm) * z

# print(my_R)

# print()

# print(tvec)

R, J = cv2.Rodrigues(or_rvec)

y = np.array([[-mtx[0][0]], [-mtx[1][1]], [-mtx[2][2]]])

print(y)

print(np.dot(R, y))


