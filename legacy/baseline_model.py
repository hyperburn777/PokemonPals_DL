###     LEGACY MODEL   ###
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    Dropout,
    Dense,
)


class CNN:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        model = Sequential()
        inputShape = (height, width, depth)
        # "channels last" and the channels dimension itself
        chanDim = -1

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # # (CONV => RELU) * 2 => POOL
        # model.add(Conv2D(128, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Conv2D(128, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(256))  # 1024
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))  # 0.5

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
