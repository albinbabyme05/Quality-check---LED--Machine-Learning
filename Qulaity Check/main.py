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
