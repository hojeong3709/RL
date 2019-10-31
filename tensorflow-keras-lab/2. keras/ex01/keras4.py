from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # original data --> scaling
    print(x_train.shape)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255
    print(x_train.shape)

    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32") / 255

    y_train = np_utils.to_categorical(y_train) # one hot encoding
    y_test = np_utils.to_categorical(y_test) # one hot encoding

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    # FC layer
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    # Output layer
    model.add(Dense(10, activation="softmax"))

    print(model.summary)

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=30,
              batch_size=200)

    print("accuracy:", model.evaluate(x_test, y_test)[1])
