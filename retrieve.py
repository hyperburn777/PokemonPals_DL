import os
import cv2
import numpy as np
import tensorflow as tf

from config import DATA_DIR, RESULT_DIR
from data import load_datasets
from hashing import Hasher, preprocess_silhouette

POKEMON = "Abomasnow"
IMG_IDX = "0"
QUERY_IMG_PATH = DATA_DIR + f"/{POKEMON}/" + IMG_IDX + ".jpg"


def model_fn(img):
    x = preprocess_silhouette(img)
    return feature_model(x, training=False).numpy()[0]


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = load_datasets(return_class_names=True)
    num_classes = len(class_names)

    assert os.path.exists(QUERY_IMG_PATH), f"Query image not found: {QUERY_IMG_PATH}"

    best_model_path = os.path.join(RESULT_DIR, "best.keras")
    model = tf.keras.models.load_model(best_model_path)

    logit_layer = model.layers[-1]
    feature_model = tf.keras.Model(model.input, logit_layer.input)
    print(f"Feature model built using layer: {logit_layer.name}")

    print("Extracting silhouettes from test set…")
    silhouettes = {}
    for batch_imgs, batch_labels in test_ds.take(50):
        # print(f"Processing batch of size {len(batch_imgs)}")
        # print(batch_labels.numpy())
        for img, label in zip(batch_imgs, batch_labels):
            cname = class_names[int(label)]
            if cname not in silhouettes:
                silhouettes[cname] = img.numpy().squeeze()
            if len(silhouettes) == num_classes:
                break
        if len(silhouettes) == num_classes:
            break

    print("Building hash index …")
    hasher = Hasher(model_fn=model_fn, Hsize=4096)
    hasher.build_index(silhouettes)

    print(f"Querying for nearest neighbors …")
    query_img = cv2.imread(QUERY_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    nbrs = hasher.query(query_img, top_k=5)
    for n in nbrs:
        print(f"Nearest: {n['id']}  sim={n['similarity']:.4f}")
          
"""
    Output:
        Nearest: Abomasnow  sim=0.8729
        Nearest: Turtonator  sim=0.6730
        Nearest: Krookodile  sim=0.5206
        Nearest: Wartortle  sim=0.5042
        Nearest: Arcanine  sim=0.4935
"""
