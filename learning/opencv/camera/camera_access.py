import sys
import cv2 as cv


video = cv.VideoCapture(0)
window = "Recording"
cv.namedWindow(window,cv.WINDOW_NORMAL)


while cv.waitKey(1) != ord("q"):
    has_frame, frame = video.read()
    if not has_frame:
        break
    
    cv.imshow(window,frame)

video.release()
cv.destroyAllWindows()