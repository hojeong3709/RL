
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import numpy as np
import tensorflow as tf
import os

tf.set_random_seed(777)

if __name__ == "__main__":
    df = pd.read_csv("./data/wine.csv", header=None)
    data = df.values

    x_data = data[:, 0:12]
    y_data = data[:, 12]

    print(x_data.shape)
    print(y_data.shape)

    model = Sequential()
    model.add(Dense(30, input_dim=12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"

    # Validation loss Monitor
    from keras.callbacks import ModelCheckpoint
    check_pointer = ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)

    # Overfit Monitor
    from keras.callbacks import EarlyStopping
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=100)

    history = model.fit(x_data, y_data,
                        validation_split=0.33,
                        epochs=3500,
                        batch_size=500,
                        callbacks=[check_pointer, early_stop_callback])

    y_vloss = history.history["val_loss"]
    y_acc = history.history["acc"]

    x_len = np.arange(len(y_acc))

    import matplotlib.pyplot as plt
    plt.plot(x_len, y_vloss, c="red", markersize=2, label="validation_loss")
    plt.plot(x_len, y_acc, c="blue", markersize=2, label="train_accuracy")
    plt.legend()
    plt.show()
    print("accuracy:", model.evaluate(x_data, y_data)[1])
