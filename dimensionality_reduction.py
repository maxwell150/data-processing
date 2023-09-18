import cv2, glob, random, os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


breeds = glob.glob("./images/*")
two_classes = random.sample(breeds, 2)

# Loading and converting images to grayscale
def load_and_convert_to_grayscale(folder):
    images = []
    for image_path in glob.glob(os.path.join(folder, '*.jpg')):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(gray_image)
    return images

class1_images = load_and_convert_to_grayscale(two_classes[0])
class2_images = load_and_convert_to_grayscale(two_classes[1])

# converting images to grayscale pixel intensity histograms
def extract_histograms(images):
    histograms = []
    for image in images:
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        histograms.append(histogram)
    return histograms

class1_histograms = extract_histograms(class1_images)
class2_histograms = extract_histograms(class2_images)

# Normalizing the dataset
scaler = StandardScaler()
class1_normalized = scaler.fit_transform(class1_histograms)
class2_normalized = scaler.transform(class2_histograms)

# performing PCA dimensionality reduction to 2 dimensions
pca = PCA(n_components=2)
class1_pca = pca.fit_transform(class1_normalized)
class2_pca = pca.transform(class2_normalized)

# plotting the 2D points with different colors for each folder
plt.figure(figsize=(10, 6))
plt.scatter(class1_pca[:, 0], class1_pca[:, 1], c='blue', label='class 1')
plt.scatter(class2_pca[:, 0], class2_pca[:, 1], c='red', label='class 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Dimensionality Reduction')
plt.legend()
plt.grid(True)
plt.show()
