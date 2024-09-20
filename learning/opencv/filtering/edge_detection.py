import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def main():

    option = 0
    #same thing I did in camera learning project
    window = "Video"
    video = cv.VideoCapture(0)
    alive = True
    while alive:
        
        ret, frame = video.read()
        if ret:
            #preview
            if option == 0:
                cv.imshow(window,frame)
            #canny
            elif option == 1:
                result = cv.Canny(frame, 100, 200)
                cv.imshow(window,result)
            #blur
            elif option == 2:
                result = cv.blur(frame, (50,50))
                cv.imshow(window,result)
            #corners
            elif option == 3:
                gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                corners = cv.goodFeaturesToTrack(gray,maxCorners=500,qualityLevel=0.2,minDistance=15,blockSize = 9)
                
                if corners is not None:
                    for p in np.float32(corners).reshape(-1, 2):
                        result = cv.rectangle(frame,(int(p[0])-1,int(p[1])+1),(int(p[0])+1,int(p[1])-1),(0,0,255),-4)
                cv.imshow(window, result)

                
        #logic for changing camera mode
        else:
            break
        key = cv.waitKey(1)
        if key == ord("q"):
            alive = False
        elif key == ord("p"):
            option = 0
        elif key == ord("c"):
            option = 1
        elif key == ord("b"):
            option = 2
        elif key == ord("f"):
            option = 3
        


if __name__ == "__main__":
    main()