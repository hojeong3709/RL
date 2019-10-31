from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
tf.set_random_seed(777)

data = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
x = data[:, 0:8]
y = data[:, 8]

print(x.shape)
print(y.shape)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(x, y, epochs=200, batch_size=10)

result = model.evaluate(x, y)
print("loss: {} accuracy: {}".format(result[0], result[1]))