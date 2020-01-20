""" from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
 """

import keras
from keras.datasets import mnist
from keras import backend as K
from keras.layers import Dense, Input, Flatten

from KerasGA.GAModel import GAModel
# from cmaes import CMA
import KerasGA.Evolutionary_Optimizers


batch_size = 128
num_classes = 10
epochs = 40000

max_epochs = 40000

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = Input(shape=(28, 28, 1))
flatten = Flatten()(inputs)
dense = Dense(64, activation="relu")(flatten)
dense = Dense(64, activation="relu")(dense)
prediction = Dense(10, activation="softmax")(dense)

model = GAModel(input_tensor=inputs, output_tensor=prediction)

myopt = Evolutionary_Optimizers.NGA(population_size=2, sigma_original=15)
model.compile(optimizer="nga", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    # validation_data=(x_test, y_test)
)
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
