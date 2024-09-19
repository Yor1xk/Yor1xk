import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

image = "image.jpg"

def main():
    cv_image = cv.imread(image, cv.IMREAD_COLOR)
    print(cv_image)
    print("Image size:", cv_image.size)
    print("Image shape:", cv_image.shape)
    #show image in bgr colour scheme
    plt.imshow(cv_image)
    plt.waitforbuttonpress(10)

    #show image in rgb colour scheme
    plt.imshow(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB))
    plt.waitforbuttonpress(10)

    #split an image into different color channels
    b,g,r = cv.split(cv_image)

    #show every channel (b,g,r)
    plt.imshow(b)
    plt.waitforbuttonpress(10)
    plt.imshow(g)
    plt.waitforbuttonpress(10)
    plt.imshow(r)
    plt.waitforbuttonpress(10)

    #merge every channel into one image, then show it again, just to show that it hasn't been changed
    merged = cv.merge((b,g,r))
    plt.imshow(merged)
    plt.waitforbuttonpress(10)
    #convert the image to the RGB colour scheme
    plt.imshow(cv.cvtColor(merged, cv.COLOR_BGR2RGB))
    plt.waitforbuttonpress(10)


if __name__ == "__main__":
    main()
