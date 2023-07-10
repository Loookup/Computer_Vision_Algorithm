import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

num = 0

while True:

    ret, frame = capture.read()

    cv2.imshow("Video", frame)
    if cv2.waitKey(10) == 27:
        break
    elif cv2.waitKey(10) == ord('c'):
        num += 1
        print("%s Cap"%num)
        img_cap = cv2.imwrite('cap' + str(num) + '.jpg', frame)

capture.release()
cv2.destroyAllWindows()