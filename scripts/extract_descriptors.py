from skimage.feature import hog
from skimage import img_as_ubyte, color

import numpy as np
import cv2


# function to extract HOG or SIFT descriptors from an image
def extract_descriptors(images, labels, method):
    if method == 'sift':
        # create the SIFT feature extractor using OpenCV
        sift = cv2.SIFT_create()
    
    # define lists to be returned later
    descriptors = []
    y_train = []
    
    # iterate through all provided images
    for i in range(len(images)):
        img = images[i]
        label = labels[i]
        
        if method == 'sift':
            # grayscale for more effective sift features extraction
            img = img_as_ubyte(color.rgb2gray(img))
            # use sift to find feature descriptors for this image
            kp, desc = sift.detectAndCompute(img, None)
        elif method == 'hog':
            # use hog to find feature descriptors
            desc = hog(img, multichannel=True)
    
        # then append to the lists
        if desc is not None:
            y_train.append(label)
            descriptors.append(desc)
    
    return descriptors, y_train