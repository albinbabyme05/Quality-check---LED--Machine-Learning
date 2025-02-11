from cv2 import cv2
import numpy as np
import os

path = 'test'
images = []
class_names = []

"""Oriented FAST and Rotated BRIEF (ORB) Feature Detector"""
orb = cv2.ORB_create(nfeatures=2000)

images_list = os.listdir(path)
for item in images_list:
    current_image = cv2.imread(f'{path}/{item}')
    images.append(current_image)
    class_names.append(os.path.splitext(item)[0])

print("Loaded images:", class_names)


def descriptFinder(images):
    """Find descriptors for each loaded image using ORB"""
    descriptor_list = []
    for image in images:
        keypoint, descriptor = orb.detectAndCompute(image, None)
        descriptor_list.append(descriptor)
    return descriptor_list

descriptor_list = descriptFinder(images=images)
print(f"Total descriptors extracted: {len(descriptor_list)}")



def cameraDescriptor(image, descriptor_list):
    """Match features from camera frame against stored descriptors"""
    camKeypoint, camDescriptor = orb.detectAndCompute(image, None)
    bf = cv2.BFMatcher()
    finalMatchList = []
    finalValue = -1  # Default value

    try:
        for descript in descriptor_list:
            matches = bf.knnMatch(descript, camDescriptor, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            finalMatchList.append(len(good_matches))
    except:
        pass

    if finalMatchList and max(finalMatchList) > 10:  # Threshold for matching
        finalValue = finalMatchList.index(max(finalMatchList))

    return finalValue
