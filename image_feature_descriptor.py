import cv2 as cv
import matplotlib.pyplot as plt
import glob, random


breed_Pekinese = glob.glob('./images/n02086079-Pekinese/*')
image_pekinese = random.choice(breed_Pekinese)

# Loading the image
image = cv.imread(image_pekinese, cv.IMREAD_GRAYSCALE) 

# Creating ORB detector with specified parameters
edge_threshold = 16
patch_size = 30

orb = cv.ORB_create(
    edgeThreshold=edge_threshold,
    patchSize=patch_size,
    nlevels=8,
    fastThreshold=20,
    scaleFactor=1.2,
    WTA_K=2,
    scoreType=cv.ORB_HARRIS_SCORE,
    firstLevel=0,
    nfeatures=30
)

#finding the keypoints with ORB
kp = orb.detect(image, None)

#drawing only keypoints location
img_with_keypoints = cv.drawKeypoints(image, kp, None, color=(0, 255, 0))

# printing the number of keypoints extracted
num_keypoints = len(kp)

# displaying the image with keypoints
plt.figure(figsize=(8, 8))
plt.imshow(cv.cvtColor(img_with_keypoints, cv.COLOR_BGR2RGB))
plt.title(f'Number of Keypoints: {num_keypoints}\nEdge Threshold: {edge_threshold}\nPatch Size: {patch_size}')
plt.axis('off')
plt.show()

# Printing the results
print(f"Number of Keypoints Extracted: {num_keypoints}")
print(f"Edge Threshold Value: {edge_threshold}")
print(f"Patch Size: {patch_size}")

