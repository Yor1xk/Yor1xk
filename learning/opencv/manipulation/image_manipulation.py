import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

image = "image.jpg"
image1 = "image1.png"

def main():
    #read the checkerboard image and the cat image
    cv_image = cv.imread(image1,0)
    cat = cv.imread(image,1)
    #print numpy representation
    print(cv_image)

    #show it on matplot
    plt.imshow(cv_image, cmap = "gray")
    plt.waitforbuttonpress(timeout=-1)

    #modify some pixels and then show the image
    cv_image[20, 20] = 0
    cv_image[20, 30] = 0
    cv_image[30, 20] = 0
    cv_image[30, 30] = 0
    plt.imshow(cv_image, cmap = "gray")
    plt.waitforbuttonpress(timeout=-1)

    #reverse the B,G,R channels
    cat = cat[:,:, ::-1]
    plt.imshow(cat)
    plt.waitforbuttonpress(-1)

    #crop the image
    cat = cat[200:400, 200:400]
    plt.imshow(cat)
    plt.waitforbuttonpress(-1)

    #resize demo
    print("Cropped cat size:", cat.size)
    print("Cropped cat shape:", cat.shape)

    cat = cv.resize(cat,(50,50))
    plt.imshow(cat)
    plt.waitforbuttonpress(-1)

    print("Resized cat size:", cat.size)
    print("Resized cat shape:", cat.shape)

    #Flip demo
    plt.figure(figsize=(18, 5))
    plt.subplot(141);plt.imshow(cv.flip(cat,0));plt.title("Horizontal Flip");
    plt.subplot(142);plt.imshow(cv.flip(cat,1));plt.title("Vertical Flip");
    plt.subplot(143);plt.imshow(cv.flip(cat,-1));plt.title("Both Flipped");
    plt.subplot(144);plt.imshow(cat);plt.title("Original");
    plt.waitforbuttonpress(-1)
    

if __name__ == "__main__":
    main()