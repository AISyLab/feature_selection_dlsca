import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras import *


def best_mlp_hw_rpoi_200_ascadf(classes, number_of_samples):
    # Best multilayer perceptron for ASCADf dataset
    # Number of points-of-interest: 200
    # Leakage model: HW
    # Number of parameters: 82209

    batch_size = 900
    tf.random.set_seed(699165)
    model = Sequential(name="best_mlp_hw_rpoi_200_ascadf")
    model.add(Dense(200, activation="relu", kernel_initializer="random_uniform", input_shape=(number_of_samples,)))
    model.add(Dense(200, activation="relu", kernel_initializer="random_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_mlp_id_rpoi_100_ascadf(classes, number_of_samples):
    # Best multilayer perceptron for ASCADf dataset
    # Number of points-of-interest: 100
    # Leakage model: ID
    # Number of parameters: 429256

    batch_size = 100
    tf.random.set_seed(917539)
    model = Sequential(name="best_mlp_id_rpoi_100_ascadf")
    model.add(Dense(500, activation="relu", kernel_initializer="he_uniform", input_shape=(number_of_samples,)))
    model.add(Dense(500, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_hw_rpoi_400_ascadf(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCADf key dataset
    # Number of points-of-interest: 400
    # Leakage model: HW
    # Number of parameters: 499533

    batch_size = 400
    tf.random.set_seed(299882)
    model = Sequential(name="best_cnn_hw_rpoi_400_ascadf")
    model.add(Conv1D(kernel_size=46, strides=23, filters=16, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(MaxPool1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=36, strides=18, filters=32, activation="selu", padding="same"))
    model.add(MaxPool1D(pool_size=8, strides=8, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=48, strides=24, filters=64, activation="selu", padding="same"))
    model.add(MaxPool1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=46, strides=23, filters=128, activation="selu", padding="same"))
    model.add(MaxPool1D(pool_size=10, strides=10, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(20, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(20, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(20, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(20, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_id_rpoi_200_ascadf(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCADf key dataset
    # Number of points-of-interest: 200
    # Leakage model: ID
    # Number of parameters: 158108

    batch_size = 400
    tf.random.set_seed(756790)
    model = Sequential(name="best_cnn_id_rpoi_200_ascadf")
    model.add(Conv1D(kernel_size=34, strides=17, filters=16, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(MaxPool1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=44, strides=22, filters=32, activation="selu", padding="same"))
    model.add(MaxPool1D(pool_size=10, strides=10, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=40, strides=20, filters=64, activation="selu", padding="same"))
    model.add(MaxPool1D(pool_size=6, strides=6, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(100, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(100, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(100, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size
