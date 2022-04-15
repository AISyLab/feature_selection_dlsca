import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *
import random


def best_cnn_hw_nopoi_7500_chesctf(classes, number_of_samples):
    # Best Convolutional Neural Network for CHESCTF key dataset
    # Number of points-of-interest: 7500
    # Leakage model: HW
    # Number of parameters: 374233

    batch_size = 400
    tf.random.set_seed(728495)
    model = Sequential(name="best_cnn_hw_rpoi_7500_chesctf")
    model.add(Conv1D(kernel_size=48, strides=24, filters=8, activation="relu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(200, activation="relu", kernel_initializer="he_uniform", kernel_regularizer=l2(0.001)))
    model.add(Dense(200, activation="relu", kernel_initializer="he_uniform", kernel_regularizer=l2(0.001)))
    model.add(Dense(200, activation="relu", kernel_initializer="he_uniform", kernel_regularizer=l2(0.001)))
    model.add(Dense(200, activation="relu", kernel_initializer="he_uniform", kernel_regularizer=l2(0.001)))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_id_nopoi_7500_chesctf(classes, number_of_samples):
    # Best Convolutional Neural Network for CHESCTF key dataset
    # Number of points-of-interest: 7500
    # Leakage model: ID
    # Number of parameters: 564436

    batch_size = 800
    tf.random.set_seed(225382)
    model = Sequential(name="best_cnn_id_rpoi_7500_chesctf")
    model.add(Conv1D(kernel_size=40, strides=20, filters=4, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(400, activation="selu", kernel_initializer="random_uniform"))
    model.add(Dense(400, activation="selu", kernel_initializer="random_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size
