import cv2 as cv

#create a video capture object to read video file
source = "race_car.mp4"
video = cv.VideoCapture(source)

width = int(video.get(3))
height = int(video.get(4))

#create video writer objects
avi = cv.VideoWriter("car_avi.avi",cv.VideoWriter_fourcc("M","J","P","G"), 10, (width, height), isColor=True)
mp4 = cv.VideoWriter("car_mp4.mp4",cv.VideoWriter_fourcc(*"mp4v"), 10, (width, height), isColor=True)

#while there are frames in the video, display them
while video.isOpened():
    #read the individual frame
    ret, frame = video.read()
    
    
    #display it
    if ret:
        #write each frame to the corresponding videos
        avi.write(frame)
        mp4.write(frame)
        cv.imshow("video", frame)
    else:
        break
    #essential! if not present the video will not start
    cv.waitKey(10)
#release the resources
video.release()
avi.release()
mp4.release()
cv.destroyAllWindows()