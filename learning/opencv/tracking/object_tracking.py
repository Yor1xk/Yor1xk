import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


#set up tracker
tracker_types = [
    "BOOSTING",
    "MIL",
    "KCF",
    "CSRT",
    "TLD",
    "MEDIANFLOW",
    "GOTURN",
    "MOSSE",
]
#convenience functions
#draw a rectangle over a frame
def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

#display a rectangle on a matplotlib plot
def displayRectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv.cvtColor(frameCopy, cv.COLOR_RGB2BGR)
    plt.imshow(frameCopy)
    plt.axis("off")

#draw text on a frame in a specific location
def drawText(frame, txt, location, color=(50, 170, 50)):
    cv.putText(frame, txt, location, cv.FONT_HERSHEY_SIMPLEX, 1, color, 3)


def main():
    #take the tracker type out of console
    tracker_type = 0
    if len(sys.argv) > 1:
        tracker_type = int(sys.argv[1])
    else:
        tracker_type = 6

    #take the tracker type and create a new tracker object accordingly
    tracker_name = tracker_types[tracker_type]
    
    if tracker_name == "BOOSTING":
        tracker = cv.legacy_TrackerBoosting.create()
    elif tracker_name == "MIL":
        tracker = cv.legacy_TrackerMIL.create()
    elif tracker_name == "KCF":
        tracker = cv.legacy_TrackerKCF.create()
    elif tracker_name == "CSRT":
        tracker = cv.TrackerCSRT.create()
    elif tracker_name == "TLD":
        tracker = cv.legacy_TrackerTLD.create()
    elif tracker_name == "MEDIANFLOW":
        tracker = cv.legacy_TrackerMedianFlow.create()
    elif tracker_name == "GOTURN":
        tracker = cv.TrackerGOTURN.create()
    else:
        tracker = cv.legacy_TrackerMOSSE.create()
    #read the video and create VideoCapture object
    video = "race_car.mp4"
    capture = cv.VideoCapture(filename=video)
    ok, first_frame = capture.read()
    if capture.isOpened():
        width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    else:
        sys.exit()

    #create a video writer object for writing the output video in
    output_video_name = "race_car_" + tracker_name + ".mp4"
    output_video = cv.VideoWriter(filename=output_video_name,
                                  fourcc=cv.VideoWriter.fourcc("x","v","i","d"),
                                  fps = 10, frameSize=(width,height))

    

    #select the initial bounding box
    bbox = cv.selectROI(first_frame,False)
    print(bbox)
    displayRectangle(first_frame, bbox)

    #initialize a tracker
    tracker.init(first_frame,bbox)

    
    while capture.isOpened():
        ret, frame = capture.read()

        if ret:
            cv.imshow("Video", frame)
            #start timer
            timer = cv.getTickCount()
            #update tracker
            ret, bbox = tracker.update(frame)
            #get fps
            fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

            #draw bbox
            if ret:
                drawRectangle(frame, bbox)
            else:
                drawText(frame, "Error", (80,140), (0,0,255))

            #display info
            drawText(frame, tracker_name + " Tracker", (80,60))
            drawText(frame, "FPS: " + str(int(fps)), (80,100))

            #write frame to video
            output_video.write(frame)
        else:
            break
        cv.waitKey(25)

        if not capture.isOpened():
            break    
        
    capture.release()
    output_video.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()