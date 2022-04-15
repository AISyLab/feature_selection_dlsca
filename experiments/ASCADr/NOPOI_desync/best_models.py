import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *


def best_cnn_hw_nopoi_25000_ascadr(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCADr key dataset
    # Number of points-of-interest: 25000
    # Leakage model: HW
    # Number of parameters: 22889

    batch_size = 700
    tf.random.set_seed(396821)
    model = Sequential(name="best_cnn_hw_rpoi_25000_ascadr")
    model.add(Conv1D(kernel_size=30, strides=15, filters=16, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=10, strides=10, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=40, strides=37, filters=32, activation="selu", padding="same"))
    model.add(AveragePooling1D(pool_size=6, strides=6, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(20, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(20, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(20, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_id_nopoi_25000_ascadr(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCADr key dataset
    # Number of points-of-interest: 25000
    # Leakage model: ID
    # Number of parameters: 90368

    batch_size = 600
    tf.random.set_seed(454812)
    model = Sequential(name="best_cnn_id_rpoi_25000_ascadr")
    model.add(Conv1D(kernel_size=50, strides=17, filters=4, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=4, strides=4, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=46, strides=27, filters=8, activation="selu", padding="same"))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=50, strides=21, filters=16, activation="selu", padding="same"))
    model.add(AveragePooling1D(pool_size=10, strides=10, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(300, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size
