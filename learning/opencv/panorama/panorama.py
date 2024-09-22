import glob
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
import math

#read images
imagefiles = glob.glob(f"boat{os.sep}*")
imagefiles.sort()

images = []
for image in imagefiles:
    im = cv.imread(image)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    images.append(im)

#show images
plot = plt.figure(figsize = (2,3))
for i in range(0,len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])

plot.show()
plot.waitforbuttonpress(-1)


#stitch them together to create panorama

stitcher = cv.Stitcher.create()
status, result = stitcher.stitch(images=images)

if status == 0:
    plot.clear()
    plt.imshow(result)
    plt.waitforbuttonpress(-1)

