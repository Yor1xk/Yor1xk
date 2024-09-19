import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = "image.jfif"

def main():
    #read and convert image to RGB
    img = cv.imread(image, 1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    #show the image
    plt.imshow(img)
    plt.title("Original")
    plt.waitforbuttonpress(-1)

    #matrix to test addition and subtraction
    matrix = np.ones(img.shape, dtype = "uint8") * 50
    print(matrix) #should be the matrix containing 50's
    img_bright = cv.add(img, matrix)
    img_darker = cv.subtract(img, matrix)
    
    
    plt.figure(figsize=[18,5])
    plt.subplot(131); plt.imshow(img_bright); plt.title("Bright")
    plt.subplot(132); plt.imshow(img_darker); plt.title("Darker")
    plt.waitforbuttonpress(-1)


    #demo for contrast and multiplication
    matrix1 = np.ones(img.shape) * 0.8
    matrix2 = np.ones(img.shape) * 1.2

    img_darker = np.uint8(cv.multiply(np.float64(img), matrix1))
    img_bright = np.uint8(cv.multiply(np.float64(img), matrix2))

    plt.figure(figsize=[18,5])
    plt.subplot(131); plt.imshow(img_bright); plt.title("High contrast")
    plt.subplot(132); plt.imshow(img_darker); plt.title("Low contrast")
    plt.waitforbuttonpress(-1)
    #the problem with the previous approach is that by multiplying the image by matrix2, some values can exceed the pre-assigned 0-255 range;
    #when it does so the pixel overflows to 0, thus creating a black patch we see in the demo
    #we can fix this issue using clipping
    img_bright = np.uint8(np.clip(cv.multiply(np.float64(img), matrix2),0,255))
    plt.subplot(133); plt.imshow(img_bright); plt.title("Bright fixed")
    plt.waitforbuttonpress(-1)

    #thresholding demo
    #thresholding is a popular technique for creating binary images
    #there are 2 different functions(classic and adaptive)
    #thresholding check if a pixel value is under a certain value of threshold: if it is, the value is set to 0; otherwise the value is set to 1

    retval, img_threshold = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
    plt.imshow(img_threshold)
    plt.title("Threshold")
    plt.waitforbuttonpress(-1)

    #adaptive threshold is used when classic thresholding techniques cannot be used(like the music sheet example)


    #bitwise operations
    #and
    rect = cv.imread("rect1.jpg", cv.IMREAD_GRAYSCALE)
    dim = rect.shape
    print(dim)
    circle = cv.imread("circle1.png", cv.IMREAD_GRAYSCALE)
    circle = cv.resize(circle,(dim[1],dim[0]))
    print(circle.shape)
    AND = cv.bitwise_and(rect,circle,mask = None)
    fig = plt.figure(figsize=(1,3))
    fig.add_subplot(1,3,1)
    plt.imshow(rect,cmap = "gray")
    plt.title("Rect")
    fig.add_subplot(1,3,2)
    plt.imshow(circle,cmap = "gray")
    plt.title("Circle")
    fig.add_subplot(1,3,3)
    plt.imshow(AND, cmap = "gray")
    plt.title("And")
    fig.show()
    plt.waitforbuttonpress(-1)

    #or, not and xor
    OR = cv.bitwise_or(rect,circle,mask = None)
    NOT = cv.bitwise_not(src=rect)
    XOR = cv.bitwise_xor(rect, circle, mask = None)
    fig = plt.figure(figsize=(1,3))
    fig.add_subplot(1,3,1)
    plt.imshow(OR,cmap = "gray")
    plt.title("Or")
    fig.add_subplot(1,3,2)
    plt.imshow(NOT,cmap = "gray")
    plt.title("Not")
    fig.add_subplot(1,3,3)
    plt.imshow(XOR, cmap = "gray")
    plt.title("Xor")
    fig.show()
    plt.waitforbuttonpress(-1)

    
    #background substitution
    logo = cv.imread("logo.jpg",cv.IMREAD_COLOR)
    background = cv.imread("background.jpg",cv.IMREAD_COLOR)
    logo = logo[:,:,::-1]
    background = background[:,:,::-1]
    #first of all we need to resize background so it has the same dimensions as the logo
    background = cv.resize(background,dsize=(logo.shape[1],logo.shape[0]), interpolation=cv.INTER_AREA)
    #then we need to take the logo as a grayscale
    logo_gray = cv.imread("logo.jpg",cv.IMREAD_GRAYSCALE)
    #threshold it + thresh_inverse
    ret, logo_gray_thresh = cv.threshold(logo_gray,127,255,cv.THRESH_BINARY)
    ret, logo_gray_inv = cv.threshold(logo_gray,127,255,cv.THRESH_BINARY_INV)
    #create the background part of the image(where we want for the background to be on the logo)
    thr_backg = cv.bitwise_and(background,background,mask = logo_gray_thresh)
    #create the foreground, excluding the background parts
    foreground = cv.add(logo, thr_backg,mask = logo_gray_inv)
    #add two images using bitwise or
    result = cv.bitwise_or(foreground,thr_backg)

    fig = plt.figure(figsize=(1,3))
    fig.add_subplot(1,3,1)
    plt.imshow(logo,cmap="gray")
    plt.title("Logo")
    fig.add_subplot(1,3,2)
    plt.imshow(background)
    plt.title("Background")
    fig.add_subplot(1,3,3)
    plt.imshow(result)
    plt.title("Result")
    fig.show()
    plt.waitforbuttonpress(-1)






if __name__ == "__main__":
    main()