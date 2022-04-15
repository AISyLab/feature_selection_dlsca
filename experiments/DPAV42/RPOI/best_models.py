import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *
import random


def best_mlp_hw_rpoi_900_dpav42(classes, number_of_samples):
    # Best multilayer perceptron for DPAV42 dataset
    # Number of points-of-interest: 900
    # Leakage model: HW
    # Number of parameters: 956009

    batch_size = 200
    tf.random.set_seed(1014790)
    model = Sequential(name="best_mlp_hw_rpoi_900_dpav42")
    model.add(Dense(500, activation="selu", kernel_initializer="he_uniform", input_shape=(number_of_samples,)))
    model.add(Dense(500, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(500, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_mlp_id_rpoi_100_dpav42(classes, number_of_samples):
    # Best multilayer perceptron for DPAV42 dataset
    # Number of points-of-interest: 100
    # Leakage model: ID
    # Number of parameters: 287956

    batch_size = 600
    tf.random.set_seed(251592)
    model = Sequential(name="best_mlp_id_rpoi_100_dpav42")
    model.add(Dense(300, activation="relu", kernel_initializer="random_uniform", input_shape=(number_of_samples,)))
    model.add(Dense(300, activation="relu", kernel_initializer="random_uniform"))
    model.add(Dense(300, activation="relu", kernel_initializer="random_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_hw_rpoi_700_dpav42(classes, number_of_samples):
    # Best Convolutional Neural Network for DPAV42 key dataset
    # Number of points-of-interest: 700
    # Leakage model: HW
    # Number of parameters: 34709

    batch_size = 300
    tf.random.set_seed(760902)
    model = Sequential(name="best_cnn_hw_rpoi_700_dpav42")
    model.add(Conv1D(kernel_size=46, strides=23, filters=12, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(MaxPool1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=36, strides=18, filters=24, activation="selu", padding="same"))
    model.add(MaxPool1D(pool_size=6, strides=6, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(100, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(100, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(100, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_id_rpoi_200_dpav42(classes, number_of_samples):
    # Best Convolutional Neural Network for DPAV42 key dataset
    # Number of points-of-interest: 200
    # Leakage model: ID
    # Number of parameters: 697112

    batch_size = 700
    tf.random.set_seed(413010)
    model = Sequential(name="best_cnn_id_rpoi_200_dpav42")
    model.add(Conv1D(kernel_size=26, strides=13, filters=12, activation="relu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=28, strides=14, filters=24, activation="relu", padding="same"))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=30, strides=15, filters=48, activation="relu", padding="same"))
    model.add(AveragePooling1D(pool_size=4, strides=4, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(500, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(500, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(500, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size
