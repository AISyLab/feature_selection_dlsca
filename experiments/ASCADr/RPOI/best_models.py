import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras import *


def best_mlp_hw_rpoi_200_ascadr(classes, number_of_samples):
    # Best multilayer perceptron for ASCADr dataset
    # Number of points-of-interest: 200
    # Leakage model: HW
    # Number of parameters: 565209

    batch_size = 500
    tf.random.set_seed(282273)
    model = Sequential(name="best_mlp_hw_rpoi_200_ascadr")
    model.add(Dense(400, activation="relu", kernel_initializer="random_uniform", input_shape=(number_of_samples,)))
    model.add(Dense(400, activation="relu", kernel_initializer="random_uniform"))
    model.add(Dense(400, activation="relu", kernel_initializer="random_uniform"))
    model.add(Dense(400, activation="relu", kernel_initializer="random_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_mlp_id_rpoi_20_ascadr(classes, number_of_samples):
    # Best multilayer perceptron for ASCADr dataset
    # Number of points-of-interest: 20
    # Leakage model: ID
    # Number of parameters: 639756

    batch_size = 300
    tf.random.set_seed(403932)
    model = Sequential(name="best_mlp_id_rpoi_20_ascadr")
    model.add(Dense(500, activation="relu", kernel_initializer="random_uniform", input_shape=(number_of_samples,)))
    model.add(Dense(500, activation="relu", kernel_initializer="random_uniform"))
    model.add(Dense(500, activation="relu", kernel_initializer="random_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_hw_rpoi_400_ascadr(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCADr key dataset
    # Number of points-of-interest: 400
    # Leakage model: HW
    # Number of parameters: 575369

    batch_size = 900
    tf.random.set_seed(914593)
    model = Sequential(name="best_cnn_hw_rpoi_400_ascadr")
    model.add(Conv1D(kernel_size=30, strides=15, filters=16, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(MaxPool1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(400, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(400, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(400, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(400, activation="selu", kernel_initializer="glorot_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_id_rpoi_30_ascadr(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCADr key dataset
    # Number of points-of-interest: 30
    # Leakage model: ID
    # Number of parameters: 636224

    batch_size = 700
    tf.random.set_seed(151335)
    model = Sequential(name="best_cnn_id_rpoi_30_ascadr")
    model.add(Conv1D(kernel_size=34, strides=17, filters=12, activation="relu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=10, strides=10, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(500, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(500, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(500, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size
