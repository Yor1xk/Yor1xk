import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

image = "image.jpg"


def main():
    #read the image and convert it to RGB scheme, then show the image
    cat = cv.imread(image,1)
    cat = cv.cvtColor(cat, cv.COLOR_BGR2RGB)
    plt.imshow(cat)
    plt.waitforbuttonpress(-1)

    #draw a white line over cat image
    cat_line = cv.line(cat.copy(),(100,100), (100,300), (50,50,255), 10)
    plt.imshow(cat_line)
    plt.waitforbuttonpress(-1)

    cat_circle = cv.circle(cat.copy(),(200,200), 50, (50,255,50), 4)
    plt.imshow(cat_circle)
    plt.waitforbuttonpress(-1)

    cat_rect = cv.rectangle(cat.copy(),(100,100),(300,300), (255,60,50), 5, cv.LINE_4)
    plt.imshow(cat_rect)
    plt.waitforbuttonpress(-1)

    cat_text0 = cv.putText(cat.copy(),"This is an orange cat0", (50,50),cv.FONT_HERSHEY_PLAIN, color=(0,0,0), thickness=5, bottomLeftOrigin = False, fontScale=3)
    cat_text1 = cv.putText(cat.copy(),"This is an orange cat1", (50,50),cv.FONT_HERSHEY_PLAIN, color=(0,0,0), thickness=5, bottomLeftOrigin = True, fontScale = 3)

    plot = plt.figure(figsize=(1,2))
    plot.add_subplot(1,2,1)
    plt.imshow(cat_text0)
    
    plot.add_subplot(1,2,2)
    plt.imshow(cat_text1)
    plot.show()
    plt.waitforbuttonpress(-1)





if __name__ == "__main__":
    main()
