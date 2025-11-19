import tensorflow as tf
from tensorflow.keras import layers as L, Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
)
from classification_models.tfkeras import Classifiers

from config import IMG_SIZE, CHANNELS, TRAINABLE_AT


class GrayscaleToRGB(L.Layer):
    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)


class EfficientNetPreprocess(L.Layer):
    def call(self, inputs):
        return preprocess_input(inputs)


def build_simple_cnn(classes):
    """Baseline CNN for silhouettes (lightweight & fast)."""
    h, w = IMG_SIZE
    inp = L.Input(shape=(h, w, CHANNELS))

    x = L.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Dropout(0.25)(x)

    x = L.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Dropout(0.25)(x)

    x = L.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Dropout(0.25)(x)

    x = L.Flatten()(x)
    x = L.Dense(256, activation="relu")(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.3)(x)

    out = L.Dense(classes, activation="softmax")(x)
    return Model(inp, out, name="baseline")


def build_efficientnet(classes, model):
    """1-channel → tile to 3ch → EfficientNetB0 backbone."""
    h, w = IMG_SIZE
    inp = L.Input(shape=(h, w, 1))
    x = GrayscaleToRGB()(inp)
    x = EfficientNetPreprocess()(x)
    params = {
        "include_top": False,
        "weights": "imagenet",
        "input_shape": (h, w, 3),
        "pooling": "avg",
    }

    if model == "b1":
        base = EfficientNetB1(**params)
    elif model == "b2":
        base = EfficientNetB2(**params)
    elif model == "b3":
        base = EfficientNetB3(**params)
    elif model == "b4":
        base = EfficientNetB4(**params)
    elif model == "b5":
        base = EfficientNetB5(**params)
    elif model == "b6":
        base = EfficientNetB6(**params)
    elif model == "b7":
        base = EfficientNetB7(**params)
    else:
        base = EfficientNetB0(**params)

    for layer in base.layers[:-TRAINABLE_AT]:
        layer.trainable = False
    for layer in base.layers[-TRAINABLE_AT:]:
        layer.trainable = True

    x = base(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(classes, activation="softmax")(x)

    print(
        f"Trainable layers: {sum([l.trainable for l in base.layers])}/{len(base.layers)}"
    )

    return Model(inp, out, name=f"effnet_{model}")


def build_resnet(classes, model):
    """1-channel → tile to 3ch → ResNet backbone."""
    h, w = IMG_SIZE
    inp = L.Input(shape=(h, w, 1))
    x = GrayscaleToRGB()(inp)
    params = {
        "include_top": False,
        "weights": "imagenet",
        "input_shape": (h, w, 3),
        "pooling": "avg",
    }

    if model == "18":
        BackboneClass, preprocess_input = Classifiers.get("resnet18")
        x = preprocess_input(x)
        base = BackboneClass(**params)
    elif model == "34":
        BackboneClass, preprocess_input = Classifiers.get("resnet34")
        x = preprocess_input(x)
        base = BackboneClass(**params)
    elif model == "50":
        x = tf.keras.applications.resnet.preprocess_input(x)
        base = tf.keras.applications.ResNet50(**params)

    # Set layer trainability
    for layer in base.layers[:-TRAINABLE_AT]:
        layer.trainable = False
    for layer in base.layers[-TRAINABLE_AT:]:
        layer.trainable = True

    # Classification head
    x = base(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(classes, activation="softmax")(x)

    print(
        f"Trainable layers: {sum([l.trainable for l in base.layers])}/{len(base.layers)}"
    )

    return Model(inp, out, name=f"resnet{model}")
