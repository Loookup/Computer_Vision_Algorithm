#! /usr/bin/env python3.9

from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
points = rng.random((30, 2))
hull = ConvexHull(points)

generators = np.array([[0.2, 0.2],
                       [0.2, 0.4],
                       [0.4, 0.4],
                       [0.4, 0.2],
                       [0.3, 0.6]])

# hull = ConvexHull(points=generators, qhull_options='QG4')
# 
# print(hull.simplices)
# 
# print(hull.good)
# 
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# for visible_facet in hull.simplices[hull.good]:
    # ax.plot(hull.points[visible_facet, 0],
            # hull.points[visible_facet, 1],
            # color='violet',
            # lw=6)
# convex_hull_plot_2d(hull, ax=ax)
# plt.show()

plt.plot(points[:, 0], points[:, 1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

print(hull.area)

print(hull.vertices)
print(hull.vertices[0])
print(hull.vertices[1])
plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')
plt.show()
