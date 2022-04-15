import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *
import random


def best_cnn_id_nopoi_3750_dpav42(classes, number_of_samples):
    # Best Convolutional Neural Network for DPAV42 key dataset
    # Number of points-of-interest: 3750
    # Leakage model: ID
    # Number of parameters: 34836

    batch_size = 600
    tf.random.set_seed(961377)
    model = Sequential(name="best_cnn_id_rpoi_3750_dpav42")
    model.add(Conv1D(kernel_size=28, strides=14, filters=4, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(MaxPool1D(pool_size=6, strides=6, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=42, strides=21, filters=8, activation="selu", padding="same"))
    model.add(MaxPool1D(pool_size=4, strides=4, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=48, strides=24, filters=16, activation="selu", padding="same"))
    model.add(MaxPool1D(pool_size=4, strides=4, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=40, strides=20, filters=32, activation="selu", padding="same"))
    model.add(MaxPool1D(pool_size=10, strides=10, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(20, activation="selu", kernel_initializer="he_uniform", kernel_regularizer=l2(5e-05)))
    model.add(Dense(20, activation="selu", kernel_initializer="he_uniform", kernel_regularizer=l2(5e-05)))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_hw_nopoi_3750_dpav42(classes, number_of_samples):
    # Best Convolutional Neural Network for DPAV42 key dataset
    # Number of points-of-interest: 3750
    # Leakage model: HW
    # Number of parameters: 72677

    batch_size = 800
    tf.random.set_seed(53256)
    model = Sequential(name="best_cnn_hw_rpoi_3750_dpav42")
    model.add(Conv1D(kernel_size=34, strides=17, filters=12, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(MaxPool1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(50, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(50, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(50, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size
