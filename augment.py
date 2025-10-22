import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)
import numpy as np
import pickle


print(tf.config.list_physical_devices("GPU"))

# Path to your main folder
data_dir = "data/testset"

# Create dataset
dataset = image_dataset_from_directory(
    data_dir,
    image_size=(128, 128),  # resize all images
    batch_size=None,  # if you want to get all images in memory
    label_mode="int",  # or "categorical" / "binary" / None
    color_mode="grayscale",
)

# Extract X (images) and y (labels)
X = []
y = []

for img, label in dataset:
    X.append(img.numpy())
    y.append(label.numpy())

X = np.array(X)
y = np.array(y)

print(X.shape, y.shape)

# Create an ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(
    rotation_range=30,  # random rotation
    width_shift_range=0.1,  # horizontal shift
    height_shift_range=0.1,  # vertical shift
    zoom_range=0.2,  # zoom in/out
    shear_range=10,  # shearing
    horizontal_flip=True,  # random horizontal flip
    fill_mode="nearest",
)

# Lists to store augmented images and labels
X_aug, y_aug = [], []

# Loop over each image in X
for i in range(len(X)):
    img = X[i].reshape((1,) + X[i].shape)  # add batch dimension

    j = 0
    for batch in datagen.flow(img, batch_size=1):
        X_aug.append(batch[0])
        y_aug.append(y[i])
        j += 1
        if j >= 8:  # generate 8 augmentations per image
            break

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)

with open("data/X_aug_y_aug.pkl", "wb") as file:
    pickle.dump((X_aug, y_aug), file)

# print("Original:", X.shape, "→ Augmented:", X_aug.shape)
