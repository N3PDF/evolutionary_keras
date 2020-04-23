""" Example implementation of evolutionary_keras
"""
import sys
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, Flatten
from evolutionary_keras.models import EvolModel
import evolutionary_keras.optimizers


batch_size = 128
num_classes = 10
dense_size = 16
epochs = 4000

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
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

inputs = Input(shape=(28, 28, 1))
flatten = Flatten()(inputs)
dense = Dense(dense_size, activation="relu")(flatten)
dense = Dense(dense_size, activation="relu")(dense)
prediction = Dense(10, activation="softmax")(dense)

model = EvolModel(inputs=inputs, outputs=prediction)

if sys.argv[-1] == "cma":
    myopt = evolutionary_keras.optimizers.CMA(population_size=5, sigma_init=15)
    epochs = 1
else:
    myopt = evolutionary_keras.optimizers.NGA(population_size=2, sigma_init=15)

print(" > Compiling the model")
model.compile(optimizer=myopt, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

print(" > Fitting")
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    # validation_data=(x_test, y_test)
)
score = model.evaluate(x=x_test, y=y_test, return_dict=True, verbose=0)

print("Test loss:", score['loss'])
print("Test accuracy:", score['accuracy'])
