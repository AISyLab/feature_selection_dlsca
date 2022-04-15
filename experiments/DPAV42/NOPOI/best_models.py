import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *
import random


def best_mlp_hw_nopoi_15000_dpav42(classes, number_of_samples):
    # Best multilayer perceptron for DPAV42 dataset
    # Number of points-of-interest: 15000
    # Leakage model: HW
    # Number of parameters: 1501009

    batch_size = 500
    tf.random.set_seed(315930)
    model = Sequential(name="best_mlp_hw_rpoi_15000_dpav42")
    model.add(Dense(100, activation="relu", kernel_initializer="glorot_uniform",
                    kernel_regularizer=l1(1e-05), input_shape=(number_of_samples,)))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_mlp_id_nopoi_3750_dpav42(classes, number_of_samples):
    # Best multilayer perceptron for DPAV42 dataset
    # Number of points-of-interest: 3750
    # Leakage model: ID
    # Number of parameters: 1603056

    batch_size = 400
    tf.random.set_seed(919906)
    model = Sequential(name="best_mlp_id_rpoi_3750_dpav42")
    model.add(Dense(400, activation="relu", kernel_initializer="he_uniform", input_shape=(number_of_samples,)))
    model.add(Dropout(0.3))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_hw_nopoi_15000_dpav42(classes, number_of_samples):
    # Best Convolutional Neural Network for DPAV42 key dataset
    # Number of points-of-interest: 15000
    # Leakage model: HW
    # Number of parameters: 578033

    batch_size = 900
    tf.random.set_seed(760413)
    model = Sequential(name="best_cnn_hw_rpoi_15000_dpav42")
    model.add(Conv1D(kernel_size=48, strides=24, filters=8, activation="relu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=8, strides=8, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(400, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(400, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(400, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_id_nopoi_15000_dpav42(classes, number_of_samples):
    # Best Convolutional Neural Network for DPAV42 key dataset
    # Number of points-of-interest: 15000
    # Leakage model: ID
    # Number of parameters: 89384

    batch_size = 400
    tf.random.set_seed(717501)
    model = Sequential(name="best_cnn_id_rpoi_15000_dpav42")
    model.add(Conv1D(kernel_size=28, strides=14, filters=4, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=50, strides=25, filters=8, activation="selu", padding="same"))
    model.add(AveragePooling1D(pool_size=4, strides=4, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=42, strides=21, filters=16, activation="selu", padding="same"))
    model.add(AveragePooling1D(pool_size=6, strides=6, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(300, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size
