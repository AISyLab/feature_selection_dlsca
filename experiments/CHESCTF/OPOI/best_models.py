import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *


def best_mlp_hw_opoi_4000_chesctf(classes, number_of_samples):
    # Best multilayer perceptron for CHESCTF dataset
    # Number of points-of-interest: 4000
    # Leakage model: HW
    # Number of parameters: 1383609

    batch_size = 100
    tf.random.set_seed(753132)
    model = Sequential(name="best_mlp_hw_rpoi_4000_chesctf")
    model.add(
        Dense(300, activation="selu", kernel_initializer="glorot_uniform", kernel_regularizer=l1(0.0001), input_shape=(number_of_samples,)))
    model.add(Dense(300, activation="selu", kernel_initializer="glorot_uniform", kernel_regularizer=l1(0.0001)))
    model.add(Dense(300, activation="selu", kernel_initializer="glorot_uniform", kernel_regularizer=l1(0.0001)))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_mlp_id_opoi_4000_chesctf(classes, number_of_samples):
    # Best multilayer perceptron for CHESCTF dataset
    # Number of points-of-interest: 4000
    # Leakage model: ID
    # Number of parameters: 213106

    batch_size = 900
    tf.random.set_seed(205653)
    model = Sequential(name="best_mlp_id_rpoi_4000_chesctf")
    model.add(Dense(50, activation="relu", kernel_initializer="he_uniform", kernel_regularizer=l2(0.005), input_shape=(number_of_samples,)))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_hw_opoi_4000_chesctf(classes, number_of_samples):
    # Best Convolutional Neural Network for CHESCTF key dataset
    # Number of points-of-interest: 4000
    # Leakage model: HW
    # Number of parameters: 666429

    batch_size = 400
    tf.random.set_seed(56002)
    model = Sequential(name="best_cnn_hw_rpoi_4000_chesctf")
    model.add(Conv1D(kernel_size=30, strides=15, filters=12, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(300, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.35))
    model.add(Dense(300, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.35))
    model.add(Dense(300, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.35))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_id_opoi_4000_chesctf(classes, number_of_samples):
    # Best Convolutional Neural Network for CHESCTF key dataset
    # Number of points-of-interest: 4000
    # Leakage model: ID
    # Number of parameters: 593780

    batch_size = 100
    tf.random.set_seed(855016)
    model = Sequential(name="best_cnn_id_rpoi_4000_chesctf")
    model.add(Conv1D(kernel_size=18, strides=8, filters=4, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(MaxPool1D(pool_size=8, strides=8, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=19, strides=4, filters=8, activation="selu", padding="same"))
    model.add(MaxPool1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=18, strides=9, filters=16, activation="selu", padding="same"))
    model.add(MaxPool1D(pool_size=6, strides=6, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(400, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(400, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(400, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(400, activation="selu", kernel_initializer="he_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size
