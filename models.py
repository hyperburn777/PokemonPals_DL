import tensorflow as tf
from tensorflow.keras import layers as L, Model
from tensorflow.keras.applications.efficientnet import preprocess_input

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


def build_efficientnet(classes):
    """1-channel → tile to 3ch → EfficientNetB0 backbone."""
    h, w = IMG_SIZE
    inp = L.Input(shape=(h, w, 1))
    x = GrayscaleToRGB()(inp)
    x = EfficientNetPreprocess()(x)

    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(h, w, 3), pooling="avg"
    )

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

    return Model(inp, out, name="effnet")


def build_resnet(classes):
    """1-channel → tile to 3ch → ResNet50 backbone."""
    h, w = IMG_SIZE
    inp = L.Input(shape=(h, w, 1))
    
    # Convert grayscale to 3-channel
    x = GrayscaleToRGB()(inp)
    
    # Apply the correct preprocessing for ResNet
    x = tf.keras.applications.resnet.preprocess_input(x)

    # Build the ResNet backbone
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(h, w, 3),
        pooling="avg"
    )

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

    return Model(inp, out, name="resnet50")


def build_resnet34(classes):
    # Pick backbone and preprocess function
    BackboneClass, preprocess_input = Classifiers.get('resnet34')

    # Shapes
    h, w = IMG_SIZE
    inp = L.Input(shape=(h, w, 1), name="grayscale_input")

    # Convert grayscale → RGB
    x = GrayscaleToRGB(name="gray2rgb")(inp)

    # Apply model-specific preprocessing
    x = preprocess_input(x)

    # Build the pretrained backbone
    base = BackboneClass(
        include_top=False,
        weights="imagenet",
        input_shape=(h, w, 3),  # RGB input
        pooling=None
    )

    # Apply layer freezing logic
    for layer in base.layers[:-TRAINABLE_AT]:
        layer.trainable = False
    for layer in base.layers[-TRAINABLE_AT:]:
        layer.trainable = True

    # Pass preprocessed image through backbone
    feats = base(x)

    # GAP + classification head
    feats = L.GlobalAveragePooling2D(name="gap")(feats)
    feats = L.Dropout(0.3, name="dropout")(feats)
    out = L.Dense(classes, activation="softmax", name="classifier")(feats)

    print(
        f"Trainable layers: {sum([l.trainable for l in base.layers])}/"
        f"{len(base.layers)}"
    )

    return Model(inp, out, name=f"resnet34")


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


def build_silhouette_cnn(classes):
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
