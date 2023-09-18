import cv2, os, random, glob
import matplotlib.pyplot as plt



breeds = glob.glob('./images/*')

selected_images = []

for folder in breeds:
    images_in_folder = glob.glob(os.path.join(folder, '*.jpg'))  
    selected_image = random.choice(images_in_folder)
    selected_images.append(selected_image)

# Plotting the images with their corresponding RGB histograms
plt.figure(figsize=(12, 4))
for i, image_path in enumerate(selected_images):
    image = cv2.imread(image_path)
    
    # Splitting the image into its RGB channels
    channels = cv2.split(image)
    colors = ('r', 'g', 'b')
    
    plt.subplot(1, 4, i + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Image {i + 1}')
    plt.axis('off')
    
    for j, color in enumerate(colors):
        histogram = cv2.calcHist([channels[j]], [0], None, [256], [0, 256])
        plt.plot(histogram, color=color, label=f'{color.upper()} channel')
    
    plt.xlabel('Intensity')
    plt.ylabel('Pixel Count')
    plt.xlim(0, 250)
    plt.ylim(0, 800)
    plt.legend()

plt.tight_layout()
plt.show()
