import numpy as np
import gzip
import cv2
import os

def load_mnist_labels(label_path):
    with gzip.open(label_path, 'rb') as f:
        # Skip the magic number and count
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_mnist_images(image_path):
    with gzip.open(image_path, 'rb') as f:
        f.read(16)  # Skip the magic number and dimensions
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    return images

def chop_and_merge(images):
    merged_images = []
    for img in images:
        strips = [img[:, 4*i:4*(i+1)] for i in range(7)]
        merged_image = np.concatenate(strips, axis=0)
        merged_images.append(merged_image)
    return np.array(merged_images)

def apply_fixed_threshold(images, threshold=1): ##NOTICE
    binarized_images = []
    for img in images:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binarized = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        binarized_images.append((binarized > 0).astype(int))
    return binarized_images

def save_to_text(files_dir, labels, images):
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)
    for index, img in enumerate(images):
        filename = os.path.join(files_dir, f"label_{labels[index]}_image_{index}.txt")
        with open(filename, 'w') as file:
            for row in img:
                file.write(''.join(map(str, row)) + '\n')

# Paths
image_path_train = r'C:\Users\97843\Desktop\MNIST\t10k-images-idx3-ubyte.gz'
label_path_train = r'C:\Users\97843\Desktop\MNIST\t10k-labels-idx1-ubyte.gz'
processed_dir = r'C:\Users\97843\Desktop\MNIST\Processed_01'

# Load data
images_train = load_mnist_images(image_path_train)
labels_train = load_mnist_labels(label_path_train)

# Process images
merged_images_train = chop_and_merge(images_train)
binary_images_train = apply_fixed_threshold(merged_images_train)

# Save processed images to text files
save_to_text(processed_dir, labels_train, binary_images_train)
