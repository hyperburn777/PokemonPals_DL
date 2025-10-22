# models.py
import tensorflow as tf
from tensorflow.keras import layers as L, Model
from config import IMG_SIZE, CHANNELS, TRAINABLE_AT


def build_simple_cnn(classes: int) -> tf.keras.Model:
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
    return Model(inp, out, name="SimpleCNN")


def build_efficientnet(classes: int) -> tf.keras.Model:
    """1-channel → tile to 3ch → EfficientNetB0 backbone."""
    h, w = IMG_SIZE
    inp = L.Input(shape=(h, w, CHANNELS))
    x3 = L.Concatenate()([inp, inp, inp])  # tile grayscale to RGB

    base = tf.keras.applications.EfficientNetB0(
        include_top=False, input_tensor=x3, pooling="avg"
    )
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= TRAINABLE_AT

    x = base.output
    x = L.Dropout(0.3)(x)
    out = L.Dense(classes, activation="softmax")(x)
    return Model(inp, out, name="EfficientNetB0_silhouette")


def _sepconv(x, f, k=3):
    x = L.SeparableConv2D(f, k, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    return L.Activation("relu")(x)


def _se_block(x, r=8):
    f = x.shape[-1]
    s = L.GlobalAveragePooling2D()(x)
    s = L.Dense(max(f // r, 8), activation="relu")(s)
    s = L.Dense(f, activation="sigmoid")(s)
    return L.Multiply()([x, L.Reshape((1, 1, f))(s)])


def build_silhouette_cnn(classes: int) -> tf.keras.Model:
    """Compact CNN tailored for shape recognition."""
    h, w = IMG_SIZE
    inp = L.Input(shape=(h, w, CHANNELS))

    x = _sepconv(inp, 32)
    x = _sepconv(x, 32)
    x = L.MaxPool2D()(x)
    x = _se_block(x)
    x = _sepconv(x, 64)
    x = _sepconv(x, 64)
    x = L.MaxPool2D()(x)
    x = _se_block(x)
    x = _sepconv(x, 96)
    x = _sepconv(x, 96)
    x = L.MaxPool2D()(x)

    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(classes, activation="softmax")(x)
    return Model(inp, out, name="SilhouetteCNN")
