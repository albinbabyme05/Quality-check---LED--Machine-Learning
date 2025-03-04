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



cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    original_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    descriptor_id = cameraDescriptor(frame, descriptor_list)
    if descriptor_id != -1:
        cv2.putText(original_frame, class_names[descriptor_id], (50, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow('Frame image', original_frame)
    key = cv2.waitKey(1) % 256
    if key == 27:  # Escape key to exit
        print("Escape pressed. Closing the application")
        break
cap.release()
cv2.destroyAllWindows()


# """Loading two test images"""
# image1 = cv2.imread('test/LED.jpg', 0)
# image2 = cv2.imread('train/switch3.jpg', 0)

# """Resizing the images"""
# image1 = cv2.resize(image1, (800, 600))
# image2 = cv2.resize(image2, (800, 600))

# """Detect and Compute ORB Features"""
# point_1, description_1 = orb.detectAndCompute(image1, None)
# point_2, description_2 = orb.detectAndCompute(image2, None)

# """Brute-Force Matching"""
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(description_1, description_2, k=2)

# good_matches = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append([m])

# print(f"Number of good matches: {len(good_matches)}")
# image3 = cv2.drawMatchesKnn(image1, point_1, image2, point_2, good_matches, None, flags=2)

# """Display images"""
# cv2.imshow('Image1', image1)
# cv2.imshow('Image2', image2)
# cv2.imshow('Matched Features', image3)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
