import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *
import random


def best_mlp_hw_opoi_800_dpav42(classes, number_of_samples):
    # Best multilayer perceptron for DPAV42 dataset
    # Number of points-of-interest: 800
    # Leakage model: HW
    # Number of parameters: 48159

    batch_size = 100
    tf.random.set_seed(1020186)
    model = Sequential(name="best_mlp_hw_rpoi_800_dpav42")
    model.add(Dense(50, activation="relu", kernel_initializer="he_uniform", input_shape=(number_of_samples,)))
    model.add(Dense(50, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(50, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(50, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.0003)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_hw_opoi_800_dpav42(classes, number_of_samples):
    # Best Convolutional Neural Network for DPAV42 key dataset
    # Number of points-of-interest: 800
    # Leakage model: HW
    # Number of parameters: 1158857

    batch_size = 50
    tf.random.set_seed(122335)
    model = Sequential(name="best_cnn_hw_rpoi_800_dpav42")
    model.add(Conv1D(kernel_size=24, strides=6, filters=12, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(MaxPool1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(500, activation="selu", kernel_initializer="random_uniform"))
    model.add(Dense(500, activation="selu", kernel_initializer="random_uniform"))
    model.add(Dense(500, activation="selu", kernel_initializer="random_uniform"))
    model.add(Dense(500, activation="selu", kernel_initializer="random_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_mlp_id_opoi_800_dpav42(classes, number_of_samples):
    # Best multilayer perceptron for DPAV42 dataset
    # Number of points-of-interest: 800
    # Leakage model: ID
    # Number of parameters: 251856

    batch_size = 200
    tf.random.set_seed(990159)
    model = Sequential(name="best_mlp_id_rpoi_800_dpav42")
    model.add(Dense(200, activation="selu", kernel_initializer="glorot_uniform", input_shape=(number_of_samples,)))
    model.add(Dense(200, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.003)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_id_opoi_800_dpav42(classes, number_of_samples):
    # Best Convolutional Neural Network for DPAV42 key dataset
    # Number of points-of-interest: 800
    # Leakage model: ID
    # Number of parameters: 424956

    batch_size = 50
    tf.random.set_seed(214474)
    model = Sequential(name="best_cnn_id_rpoi_800_dpav42")
    model.add(Conv1D(kernel_size=20, strides=6, filters=12, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(400, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.003)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size
