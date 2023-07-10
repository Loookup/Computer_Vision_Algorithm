import numpy as np
import cv2

img = cv2.imread('Coin.jpg', 0)

ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), dtype=np.uint8)

# Erosion
# To Eliminate Noise and detach Objects
Result1 = cv2.erode(img, kernel, iterations=1)

# Dilation
# De-Erosion or attach Objects and make into one
Result2 = cv2.dilate(img, kernel, iterations=1)

# Opening
# Erosion -> Dilation
# To Eliminate Noise
Result3 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

# Closing
# Dilation -> Erosion
# To cover holes
Result4 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

# img_noise = cv2.imread('letter_j_noise.png')
# Result = cv2.morphologyEx(img_noise, cv2.MORPH_OPEN, kernel)
# cv2.imshow("Source_Noise", img_noise)
# cv2.imshow("Non-Noise", Result)

cv2.imshow("Source", img)
cv2.imshow("Result_Erode", Result1)
cv2.imshow("Result_Dilate", Result2)
cv2.imshow("Result_Opening", Result3)
cv2.imshow("Result_Closing", Result4)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
