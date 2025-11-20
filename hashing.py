"""
Hash-table-based nearest-neighbor retrieval for transformed Pokemon silhouettes.
Extended from the method in:
@INPROCEEDINGS{5597515,
  author={Ando, Hiroaki and Fujiyoshi, Hironobu},
  booktitle={2010 20th International Conference on Pattern Recognition},
  title={Human-Area Segmentation by Selecting Similar Silhouette Images Based on Weak-Classifier Response},
  year={2010},
  volume={},
  number={},
  pages={3444-3447},
  keywords={Humans;Image segmentation;Accuracy;Feature extraction;Shape;Detectors;Training;Object detection and recognition},
  doi={10.1109/ICPR.2010.841}}

- Convert the score vector to a binary vector u via u_j = 1 if h_j >= 0 else 0.
- Compute hash index Hindex = (sum_j u_j * 2^j) mod Hsize.
- Store (id, score) in a hash table under this index.
- At query time, do the same for a new silhouette and compare by cosine similarity
  to all silhouettes in that bucket.
"""

import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

from config import IMG_SIZE, CHANNELS
from models import build_efficientnet


class Hasher:
    """
    Hash-table based nearest-neighbor retrieval over silhouette score vectors.

    - model_fn: callable that maps raw img -> 1D score vector (T,)
    - Hsize: number of hash buckets
    """

    def __init__(self, model_fn, Hsize=4096):
        self.model_fn = model_fn
        self.Hsize = Hsize
        self.hash_table = {}  # idx -> list of { "id": key, "score": np.ndarray (T,) }

    @staticmethod
    def _binarize(score_vec):
        """u_j = 1 if h_j >= 0 else 0 (Eq. 2)."""
        return (score_vec >= 0).astype(np.uint8)

    def _hash_index(self, u):
        """Hindex = (sum_j u_j * 2^j) mod Hsize (Eq. 3)."""
        idx = 0
        H = self.Hsize
        for j, bit in enumerate(u):
            if bit:
                idx = (idx + pow(2, j, H)) % H
        return idx

    def add(self, key, img):
        """
        key: identifier (e.g. Pokémon name or ID)
        img: silhouette image (np.ndarray), black on white, shape ≈ IMG_SIZE
        """
        score = self.model_fn(img)
        u = self._binarize(score)
        idx = self._hash_index(u)

        if idx not in self.hash_table:
            self.hash_table[idx] = []

        self.hash_table[idx].append({"id": key, "score": score})

    def build_index(self, images_dict):
        """
        images_dict: dict[key -> np.ndarray image]
        """
        for key, img in images_dict.items():
            self.add(key, img)

    def query(self, img_query, top_k=5, fallback_global=True):
        """
        Returns top_k nearest neighbors by cosine similarity in the bucket
        with the same hash index.
        """
        x = self.model_fn(img_query)
        u = self._binarize(x)
        idx = self._hash_index(u)

        if idx in self.hash_table and len(self.hash_table[idx]) > 0:
            candidates = self.hash_table[idx]
            # print(f"Using hash bucket {idx} with {len(candidates)} candidates")
        elif fallback_global:
            # Flatten all buckets into one list
            candidates = []
            for bucket in self.hash_table.values():
                candidates.extend(bucket)
            # print(f"Bucket {idx} empty → using global fallback over {len(candidates)} silhouettes")
            if not candidates:
                return []
        else:
            # strict mode: no fallback, just return empty
            return []

        Y = np.stack([c["score"] for c in candidates], axis=0)
        X = x.reshape(1, -1)

        sims = cosine_similarity(X, Y)[0]
        order = np.argsort(-sims)

        nbrs = []
        for i in order[:top_k]:
            nbrs.append(
                {
                    "id": candidates[i]["id"],
                    "similarity": float(sims[i]),
                    "score": candidates[i]["score"],
                }
            )
        return nbrs


def preprocess_silhouette(img):
    """
    Make a single silhouette image compatible with your training config.
    - img: np.ndarray (H, W) or (H, W, 3)
    - uses IMG_SIZE and CHANNELS from config
    """
    if CHANNELS == 1:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    img = img.astype("float32")
    if img.max() > 1.5:  # assume 0–255 → scale to 0–1
        img = img / 255.0

    if CHANNELS == 1:
        if img.ndim == 2:
            img = img[..., np.newaxis]  # (H, W, 1)
    else:
        pass

    return np.expand_dims(img, axis=0)  # (1, H, W, C)


def build_feature_model(classes):
    """
    Returns:
        feature_model: Keras Model that maps (H,W,C) -> 1D feature vector.
    Uses your existing model builders from models.py.
    """
    classifier = build_efficientnet(classes, model="b0")
    logit_layer = classifier.layers[-1]
    feature_model = tf.keras.Model(
        classifier.input,
        logit_layer.input,
        name="effnet_feat",
    )

    return classifier, feature_model


def make_model_fn(feature_model):
    """
    Wraps a Keras feature_model into a callable img -> 1D score vector.
    """

    def _model_fn(img):
        x = preprocess_silhouette(img)
        return feature_model(x, training=False).numpy()[0]

    return _model_fn
