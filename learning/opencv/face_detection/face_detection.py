import os
import cv2 as cv
import sys

#this topic is quite complex, so i simply tried following the code on opencv course and understand what the following lines mean.
def main():
    #
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    video = cv.VideoCapture(source)
    win_name = "Window"
    cv.namedWindow(winname=win_name, flags=cv.WINDOW_NORMAL)

    #read a preloaded nn
    neural_network = cv.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

    #model params
    in_width = 300
    in_height = 300
    mean = [104,117,123]
    conf_threshold = 0.7


    while cv.waitKey(1) != 27:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        #create a 4d blob from a frame
        blob = cv.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB = False, crop = False)

        #run a model
        neural_network.setInput(blob)
        detections = neural_network.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > conf_threshold:
                #last 4 entries of the last dimension of the blob are positions of 4 angles of a detection
                x_left_bottom = int(detections[0,0,i,3] * frame_width)
                y_left_bottom = int(detections[0,0,i,4] * frame_height)
                x_right_top = int(detections[0,0,i,5] * frame_width)
                y_right_top = int(detections[0,0,i,6] * frame_height)

                #then we just draw a capturing rect
                cv.rectangle(frame, (x_left_bottom,y_left_bottom), (x_right_top, y_right_top), (0,255,0))

                #same thing for labels(first we draw a white rectangle under text so it is more readable)
                label = "Confidence: %.4f" % confidence
                label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv.rectangle(frame,
                             (x_left_bottom, y_left_bottom - label_size[1]),
                             (x_left_bottom + label_size[0], y_left_bottom + base_line),
                             (255,255,255),
                             cv.FILLED
                             )
                cv.putText(frame,label,(x_left_bottom, y_left_bottom), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

                #get some info on nets performance and normalize it
                t, _ = neural_network.getPerfProfile()
                label = "Inference time: %.2f ms" % (t * 1000.0 / cv.getTickFrequency())
                #then write it on the frame
                cv.putText(frame, label, (0,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
                
        cv.imshow(win_name, frame)



    video.release()
    cv.destroyWindow(win_name)       


if __name__ == "__main__":
    main()
