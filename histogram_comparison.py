import cv2, glob, random
import numpy as np


breed_chihuahua = glob.glob('./images/n02085620-Chihuahua/*')

# Loading 2 images from the same folder and 1 image from another folder
two_images_chihuahua = random.sample(breed_chihuahua, 2)

# selecting a random image from the breed Pekinese
breed_Pekinese = glob.glob('./images/n02086079-Pekinese/*')
image_pekinese = random.choice(breed_Pekinese)


# Converting images to grayscale and compute histograms
grayscale_histograms = []

for image_path in [two_images_chihuahua[0], two_images_chihuahua[1], image_pekinese]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    grayscale_histograms.append(histogram)

# Defining histogram comparison metrics/functions
def euclidean_distance(hist1, hist2):
    return np.linalg.norm(hist1 - hist2)

def manhattan_distance(hist1, hist2):
    return np.abs(hist1 - hist2).sum()

def bhattacharyya_distance(hist1, hist2):
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    return -np.log(np.sum(np.sqrt(hist1 * hist2)))

def histogram_intersection(hist1, hist2):
    return np.minimum(hist1, hist2).sum()

# Comparing histograms between the images
same_class_distance_1 = euclidean_distance(grayscale_histograms[0], grayscale_histograms[1])
same_class_distance_2 = manhattan_distance(grayscale_histograms[0], grayscale_histograms[1])
same_class_distance_3 = bhattacharyya_distance(grayscale_histograms[0], grayscale_histograms[1])
same_class_distance_4 = histogram_intersection(grayscale_histograms[0], grayscale_histograms[1])

different_class_distance_1 = euclidean_distance(grayscale_histograms[0], grayscale_histograms[2])
different_class_distance_2 = manhattan_distance(grayscale_histograms[0], grayscale_histograms[2])
different_class_distance_3 = bhattacharyya_distance(grayscale_histograms[0], grayscale_histograms[2])
different_class_distance_4 = histogram_intersection(grayscale_histograms[0], grayscale_histograms[2])

# Step 5: Print and analyze the distances
print("Distance metrics between images from the same class:")
print(f"Euclidean Distance: {same_class_distance_1}")
print(f"Manhattan Distance: {same_class_distance_2}")
print(f"Bhattacharyya Distance: {same_class_distance_3}")
print(f"Histogram Intersection: {same_class_distance_4}")

print("\nDistance metrics between images from different classes:")
print(f"Euclidean Distance: {different_class_distance_1}")
print(f"Manhattan Distance: {different_class_distance_2}")
print(f"Bhattacharyya Distance: {different_class_distance_3}")
print(f"Histogram Intersection: {different_class_distance_4}")
