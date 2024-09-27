import numpy as np
import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt


FONTFACE = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

classFile  = "coco_class_labels.txt"
with open(classFile) as fp:
    labels = fp.read().split("\n")
    
print(labels)

# For each file in the directory
def detect_objects(net, im, dim = 300):

    # Create a blob from the image
    blob = cv.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)

    # Pass blob to the network
    net.setInput(blob)

    # Peform Prediction
    objects = net.forward()
    return objects

def display_text(im, text, x, y):
    # Get text size
    textSize = cv.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]

    # Use text size to create a black rectangle
    cv.rectangle(
        im,
        (x, y - dim[1] - baseline),
        (x + dim[0], y + baseline),
        (0, 0, 0),
        cv.FILLED,
    )

    # Display text inside the rectangle
    cv.putText(
        im,
        text,
        (x, y - 5),
        FONTFACE,
        FONT_SCALE,
        (0, 255, 255),
        THICKNESS,
        cv.LINE_AA,
    )



def display_objects(im, objects, threshold=0.25):
    rows = im.shape[0]
    cols = im.shape[1]

    # For every Detected Object
    for i in range(objects.shape[2]):
        # Find the class and confidence
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Recover original cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        # Check if the detection is of good quality
        if score > threshold:
            display_text(im, "{}".format(labels[classId]), x, y)
            cv.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Convert Image to RGB since we are using Matplotlib for displaying image
    mp_img = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(30, 10))
    plt.imshow(mp_img)
    plt.show()


#display object on a frame
def display_objects_frame(im, objects, threshold=0.25):
    rows = im.shape[0]
    cols = im.shape[1]

    # For every Detected Object
    for i in range(objects.shape[2]):
        # Find the class and confidence
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Recover original cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        # Check if the detection is of good quality
        if score > threshold:
            display_text(im, "{}".format(labels[classId]), x, y)
            cv.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv.imshow("Video",im)


def main():
    #read the model
    modelFile  = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
    configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")


    # Read the Tensorflow network
    net = cv.dnn.readNetFromTensorflow(modelFile, configFile)


    video = cv.VideoCapture(0)

    #detect objects in runtime
    while True:
        if cv.waitKey(1) == ord("q"):
            break
        ret, frame = video.read()


        if not ret:
            break

        

        objects = detect_objects(net,frame)
        display_objects_frame(frame,objects)



    video.release()
    cv.destroyAllWindows()

    #object detection for images included in project folder
    """
    #show results
    im = cv.imread(os.path.join("images", "street.jpg"))
    objects = detect_objects(net, im)
    display_objects(im, objects)
    

    im = cv.imread(os.path.join("images", "baseball.jpg"))
    objects = detect_objects(net, im)
    display_objects(im, objects, 0.2)

    im = cv.imread(os.path.join("images", "soccer.jpg"))
    objects = detect_objects(net, im)
    display_objects(im, objects)
    """
    

if __name__ == "__main__":
    main()