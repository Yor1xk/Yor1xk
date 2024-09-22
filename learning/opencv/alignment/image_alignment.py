import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def main():
    #read reference and alignment images
    r = "ref.jpg"
    a = "align.jpg"
    ref = cv.imread(r, cv.IMREAD_COLOR)
    align = cv.imread(a, cv.IMREAD_COLOR)

    ref = cv.cvtColor(ref,cv.COLOR_BGR2RGB)
    align = cv.cvtColor(align, cv.COLOR_BGR2RGB)

    ref_gray = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
    align_gray = cv.cvtColor(align, cv.COLOR_BGR2GRAY)

    #create orb object(feature detection)
    NUMBERS_FEATURES = 500
    orb = cv.ORB.create(nfeatures=NUMBERS_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(ref_gray,None)
    keypoints2, descriptors2 = orb.detectAndCompute(align_gray,None)

    #draw keypoints found by detect and compute method
    display1 = cv.drawKeypoints(ref, keypoints1, outImage=np.array([]), color = (255,0,0), flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    display2 = cv.drawKeypoints(align, keypoints2, outImage=np.array([]), color = (255,0,0), flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #display found kpoints
    plt.imshow(display1)
    plt.waitforbuttonpress(-1)
    plt.imshow(display2)
    plt.waitforbuttonpress(-1)

    #create a matcher object
    matcher = cv.DescriptorMatcher.create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    #find matches of 2 different images (reference and align); convert the result to list
    matches = list(matcher.match(descriptors1, descriptors2, None))

    #sort matches
    matches.sort(key = lambda x: x.distance, reverse = False)

    # remove not so good matches
    numGoodMatches = int(len(matches) * 0.1)
    matches = matches[:numGoodMatches]

    #draw matches
    im_matches = cv.drawMatches(ref,keypoints1, align, keypoints2, matches, None)
    plt.imshow(im_matches)
    plt.waitforbuttonpress(-1)

    #find homography
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    
    h, mask = cv.findHomography(points2, points1, cv.RANSAC)

    #warp the image
    height, width, channels = ref.shape
    warped = cv.warpPerspective(align, h, (width,height))

    plt.imshow(warped)
    plt.waitforbuttonpress(-1)



    





if __name__ == "__main__":
    main()