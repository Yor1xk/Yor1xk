import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

#I used the course's template to create a realtime pose estimation solution, but it is really slow. Be aware of this

#The link to download the model file and other smaller files that could be missing from this repo
#URL = r"https://www.dropbox.com/s/089r2yg6aao858l/opencv_bootcamp_assets_NB14.zip?dl=1"

def main():
    #two of the fundamental files for a Coffee model: model file and weights file
    protoFile   = "pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = os.path.join("model", "pose_iter_160000.caffemodel")


    #a list which describes which point(on pose estim.) should be connected to which point
    nPoints = 15
    POSE_PAIRS = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [1, 5],
        [5, 6],
        [6, 7],
        [1, 14],
        [14, 8],
        [8, 9],
        [9, 10],
        [14, 11],
        [11, 12],
        [12, 13],
    ]

    #create a nn from files
    net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)

    #set nn to cuda compiling
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    #read image
    video = cv.VideoCapture(0)
    winname = "Video"


    while cv.waitKey(1) != ord("q"):
        ret, im = video.read()
        if not ret:
            break


    
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

        inWidth  = im.shape[1]
        inHeight = im.shape[0]

        #convert image to a blob
        netInputSize = (368, 368)
        inpBlob = cv.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
        net.setInput(inpBlob)

        #make a prediction based on blob
        output = net.forward()
        """
        THIS PIECE IS FOR STATIC IMAGES

        #display probability maps
        plt.figure(figsize=(20, 5))
        for i in range(nPoints):
            probMap = output[0, i, :, :]
            displayMap = cv.resize(probMap, (inWidth, inHeight), cv.INTER_LINEAR)
            
            plt.subplot(2, 8, i + 1)
            plt.axis("off")
            plt.imshow(displayMap, cmap="jet")

        plt.waitforbuttonpress(-1)
        """
        

        #EXTRACT POINTS FROM PROB MAPS

        #get scales for X and Y axis
        scaleX = inWidth  / output.shape[3]
        scaleY = inHeight / output.shape[2]

        #here extracted points will be stored
        points = []

        #threshold value
        threshold = 0.1

        for i in range(nPoints):
            #obtain prob map from nn's output blob
            probMap = output[0, i, :, :]

            #find global maxima of the probMap.
            minVal, prob, minLoc, point = cv.minMaxLoc(probMap)

            #scale the point to fit on the original image
            x = scaleX * point[0]
            y = scaleY * point[1]

            if prob > threshold:
                #add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)


        #DISPLAY POINTS AND SKELETON
                
        imPoints = im.copy()
        imSkeleton = im.copy()

        #draw points
        for i, p in enumerate(points):
            #draws circles and numbers over detected points
            cv.circle(imPoints, p, 8, (255, 255, 0), thickness=-1, lineType=cv.FILLED)
            cv.putText(imPoints, "{}".format(i), p, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, lineType=cv.LINE_AA)

        #draw skeleton
        for pair in POSE_PAIRS:
            #takes two points, 
            partA = pair[0]
            partB = pair[1]

            #check if those two parts are in the POINTS list.
            if points[partA] and points[partB]:
                cv.line(imSkeleton, points[partA], points[partB], (255, 255, 0), 2)
                cv.circle(imSkeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv.FILLED)

        #show the result

        """
        TESTING PURPOSES

        plt.figure(figsize=(50, 50))

        plt.subplot(121)
        plt.axis("off")
        plt.imshow(imPoints)

        plt.subplot(122)
        plt.axis("off")
        plt.imshow(imSkeleton)

        plt.waitforbuttonpress(-1)
        """
        cv.imshow(winname,imSkeleton[:,:,::-1])
        #cv.imshow(winname + "1", imPoints)

    
    cv.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    main()
    