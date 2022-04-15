import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *


def best_cnn_hw_nopoi_10000_ascadf(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCADf key dataset
    # Number of points-of-interest: 10000
    # Leakage model: HW
    # Number of parameters: 268433

    batch_size = 1000
    tf.random.set_seed(1042584)
    model = Sequential(name="best_cnn_hw_rpoi_10000_ascadf")
    model.add(Conv1D(kernel_size=28, strides=14, filters=4, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=6, strides=6, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=42, strides=21, filters=8, activation="selu", padding="same"))
    model.add(AveragePooling1D(pool_size=4, strides=4, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=26, strides=13, filters=16, activation="selu", padding="same"))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(500, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(500, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_id_nopoi_10000_ascadf(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCADf key dataset
    # Number of points-of-interest: 10000
    # Leakage model: ID
    # Number of parameters: 64002

    batch_size = 1000
    tf.random.set_seed(702806)
    model = Sequential(name="best_cnn_id_rpoi_10000_ascadf")
    model.add(Conv1D(kernel_size=44, strides=22, filters=4, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(50, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(50, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(50, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size
