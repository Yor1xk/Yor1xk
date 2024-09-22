import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
#read images and create a list of corresponding exposure times
#list of file names
filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]

#list of exposure times
times = np.array([1 / 30.0, 0.25, 2.5, 15.0], dtype=np.float32)

#read images
images = []
for filename in filenames:
    im = cv.imread(filename)
    images.append(im)


#align images. If not done there will be artifacts present on final HDR image.
#the technique used is MTB
align = cv.createAlignMTB()
align.process(images, images)


#find response camera functions so then we can combine all of the images in a HDR image correctly
#we need to do this since cameras are not linear.
calibrateDebevec = cv.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)

#draw functions
x = np.arange(256, dtype=np.uint8)
y = np.squeeze(responseDebevec)

ax = plt.figure(figsize = (30,10))
plt.title("Debevec Camera Response Functions")
plt.xlabel("Measured Pixel Value")
plt.ylabel("Calibrated intensity")
plt.xlim([0,260])
plt.grid()
plt.plot(x,y[:,0], "b", x, y[:, 1], "g", x, y[:, 2], "r")
plt.waitforbuttonpress(-1)

#merge images into an HDR image
mergeDebevec = cv.createMergeDebevec()
hdr = mergeDebevec.process(images,times,responseDebevec)
hdrRGB = cv.cvtColor(hdr, cv.COLOR_BGR2RGB)

plt.clf()
plt.imshow(hdrRGB)
plt.waitforbuttonpress(-1)

#the result is that esthetically pleasing. thats why we also need to perform a tonemapping.
#there are different tonemapping techniques. Opencv course, which I followed, offers three of them: Drago, Mantiuk and Reinhard
#In this file I will only implement one of them, since doing all three seems redundant to me.

tonemapDrago = cv.createTonemapDrago(1.0, 0.7)
ldrDrago = tonemapDrago.process(hdrRGB)
ldrDrago = 3 * hdr
cv.imwrite("ldr-Drago.jpg", ldrDrago*255)

#plotting the image
plt.figure(figsize=(20, 10));plt.imshow(np.clip(ldrDrago, 0, 1)[:,:,::-1]);plt.axis("off");





