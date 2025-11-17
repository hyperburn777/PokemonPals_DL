from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle


# print(tf.config.list_physical_devices("GPU"))

DATA = "data/testset"

dataset = image_dataset_from_directory(
    DATA,
    image_size=(128, 128),  # resize
    batch_size=None,
    label_mode="int",
    color_mode="grayscale",
)

X = np.array([img for img, _ in dataset])
y = np.array([label for _, label in dataset])

print(X.shape, y.shape)

datagen = ImageDataGenerator(
    rotation_range=30,  # random rotation
    width_shift_range=0.1,  # horizontal shift
    height_shift_range=0.1,  # vertical shift
    zoom_range=0.2,  # zoom in/out
    shear_range=10,  # shearing
    horizontal_flip=True,  # random horizontal flip
    fill_mode="nearest",
)

X_aug, y_aug = [], []

# Loop over each image in X
for i in range(len(X)):
    img = X[i].reshape((1,) + X[i].shape)  # add batch dimension

    j = 0
    for batch in datagen.flow(img, batch_size=1):
        X_aug.append(batch[0])
        y_aug.append(y[i])
        j += 1
        if j >= 16:  # generate 16 augmentations per image
            break

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)

with open("data/X_aug_y_aug.pkl", "wb") as file:
    pickle.dump((X_aug, y_aug), file)

# print("Original:", X.shape, "→ Augmented:", X_aug.shape)
