import cv2
import time


def putText(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 255, 0)
    text_color_bg = (0, 0, 0)

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    offset = 5

    cv2.rectangle(img, (x-offset, y-offset), (x+text_w+offset, y+text_h+offset), text_color_bg, -1)
    cv2.putText(img, text, (x, y+text_h + font_scale - 1), font, font_scale, text_color, font_thickness)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = cap.get(cv2.CAP_PROP_FPS)
print('fps', fps)

if fps == 0.0:
    fps = 30.0


time_per_frame_video = 1/fps
last_time = time.perf_counter()

while True:

    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret1, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

    # print(frame)

    # time_per_frame = time.perf_counter() - last_time
    # time_sleep_frame = max(0, time_per_frame_video - time_per_frame)
    # time.sleep(time_sleep_frame)
    #
    # real_fps = 1/(time.perf_counter() - last_time)
    # last_time = time.perf_counter()
    #
    # if(fps < 0 or fps > 30):
    #     print(fps)
    #
    #
    # x = 30
    # y = 50
    # text = '%5f fps'%real_fps
    #
    # putText(frame, text, x, y)
    cv2.imshow("Video", frame)

    if cv2.waitKey(10) == 27:
        break


cap.release()
cv2.destroyAllWindows()

