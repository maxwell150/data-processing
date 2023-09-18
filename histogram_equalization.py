import cv2, os, cv2, glob, random
import matplotlib.pyplot as plt


breeds = glob.glob('./images/*')

selected_images = []

for folder in breeds:
    images_in_folder = glob.glob(os.path.join(folder, '*.jpg'))
    selected_images.extend(random.sample(images_in_folder, 2))

# Converting color images to grayscale using iteration
grayscale_images = []
for image_path in selected_images:
    color_image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    grayscale_images.append(grayscale_image)

#Plotting the 8 grayscale images with their histograms
plt.figure(figsize=(12, 8))
for i in range(8):
    plt.subplot(4, 4, i + 1)
    plt.imshow(grayscale_images[i], cmap='gray')
    plt.title(f'Grayscale {i + 1}')
    plt.axis('off')

    plt.subplot(4, 4, i + 9)
    plt.hist(grayscale_images[i].ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
    plt.title(f'Histogram {i + 1}')
    plt.xlim([0, 256])

plt.tight_layout()

#Performing histogram equalization on the 8 images
equalized_images = []
for grayscale_image in grayscale_images:
    equalized_image = cv2.equalizeHist(grayscale_image)
    equalized_images.append(equalized_image)

# Plotting the equalized images with their histograms
plt.figure(figsize=(12, 8))
for i in range(8):
    plt.subplot(4, 4, i + 1)
    plt.imshow(equalized_images[i], cmap='gray')
    plt.title(f'Equalized {i + 1}')
    plt.axis('off')

    plt.subplot(4, 4, i + 9)
    plt.hist(equalized_images[i].ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
    plt.title(f'Equalized Histogram {i + 1}')
    plt.xlim([0, 256])

plt.tight_layout()

# a grayscale image and its corresponding equalized image
random_index = random.randint(0, len(selected_images) - 1)
chosen_grayscale = grayscale_images[random_index]
chosen_equalized = equalized_images[random_index]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(chosen_grayscale, cmap='gray')
plt.title('Chosen Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(chosen_equalized, cmap='gray')
plt.title('Corresponding Equalized Image')
plt.axis('off')

plt.tight_layout()

plt.show()
