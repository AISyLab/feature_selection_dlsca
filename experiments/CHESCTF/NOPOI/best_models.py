import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *
import random


# def best_mlp_hw_nopoi_15000_chesctf(classes, number_of_samples):
#     # Best multilayer perceptron for CHES CTF dataset
#     # Number of points-of-interest: 15000
#     # Leakage model: HW
#     # POI interval: [0, 150000]
#     # Number of parameters: 5243209
#
#     batch_size = 300
#     seed = 567322
#     tf.random.set_seed(seed)
#     model = Sequential(name="best_mlp_hw_nopoi_chesctf_15000")
#     model.add(Dense(100, activation='relu', kernel_regularizer=l1(l=0.001), input_shape=(number_of_samples,)))
#     model.add(Dense(100, activation='relu', kernel_regularizer=l1(l=0.001)))
#     model.add(Dense(classes, activation='softmax'))
#     model.summary()
#     optimizer = Adam(lr=0.0001)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model, batch_size, seed
#
#
# def best_mlp_id_nopoi_15000_chesctf(classes, number_of_samples):
#     # Best multilayer perceptron for CHES CTF dataset
#     # Number of points-of-interest: 15000
#     # Leakage model: HW
#     # POI interval: [0, 150000]
#     # Number of parameters: 5243209
#
#     batch_size = 50
#     seed = 567322
#     tf.random.set_seed(seed)
#     model = Sequential(name="best_mlp_id_nopoi_chesctf_15000")
#     model.add(Dense(100, activation='relu', kernel_regularizer=l1(l=0.0003), input_shape=(number_of_samples,)))
#     model.add(Dense(100, activation='relu', kernel_regularizer=l1(l=0.0003)))
#     model.add(Dense(classes, activation='softmax'))
#     model.summary()
#     optimizer = Adam(lr=0.0001)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model, batch_size, seed

def best_mlp_hw_nopoi_3750_chesctf(classes, number_of_samples):
    # Best multilayer perceptron for CHESCTF dataset
    # Number of points-of-interest: 3750
    # Leakage model: HW
    # Number of parameters: 198209

    batch_size = 800
    tf.random.set_seed(410593)
    model = Sequential(name="best_mlp_hw_rpoi_3750_chesctf")
    model.add(
        Dense(50, activation="relu", kernel_initializer="glorot_uniform", kernel_regularizer=l2(0.001), input_shape=(number_of_samples,)))
    model.add(Dense(50, activation="relu", kernel_initializer="glorot_uniform", kernel_regularizer=l2(0.001)))
    model.add(Dense(50, activation="relu", kernel_initializer="glorot_uniform", kernel_regularizer=l2(0.001)))
    model.add(Dense(50, activation="relu", kernel_initializer="glorot_uniform", kernel_regularizer=l2(0.001)))
    model.add(Dense(50, activation="relu", kernel_initializer="glorot_uniform", kernel_regularizer=l2(0.001)))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_mlp_id_nopoi_30000_chesctf(classes, number_of_samples):
    # Best multilayer perceptron for CHESCTF dataset
    # Number of points-of-interest: 30000
    # Leakage model: ID
    # Number of parameters: 6091856

    batch_size = 300
    tf.random.set_seed(311851)
    model = Sequential(name="best_mlp_id_rpoi_30000_chesctf")
    model.add(Dense(200, activation="selu", kernel_initializer="random_uniform", kernel_regularizer=l1(0.0001),
                    input_shape=(number_of_samples,)))
    model.add(Dense(200, activation="selu", kernel_initializer="random_uniform", kernel_regularizer=l1(0.0001)))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_id_nopoi_7500_chesctf(classes, number_of_samples):
    # Best Convolutional Neural Network for CHESCTF key dataset
    # Number of points-of-interest: 7500
    # Leakage model: ID
    # Number of parameters: 1319552

    batch_size = 700
    tf.random.set_seed(918429)
    model = Sequential(name="best_cnn_id_rpoi_7500_chesctf")
    model.add(Conv1D(kernel_size=32, strides=16, filters=8, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(500, activation="selu", kernel_initializer="random_uniform"))
    model.add(Dense(500, activation="selu", kernel_initializer="random_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size


def best_cnn_hw_nopoi_7500_chesctf(classes, number_of_samples):
    # Best Convolutional Neural Network for CHESCTF key dataset
    # Number of points-of-interest: 7500
    # Leakage model: HW
    # Number of parameters: 216673

    batch_size = 1000
    tf.random.set_seed(203698)
    model = Sequential(name="best_cnn_hw_rpoi_7500_chesctf")
    model.add(Conv1D(kernel_size=28, strides=14, filters=8, activation="relu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(MaxPool1D(pool_size=4, strides=4, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(200, activation="relu", kernel_initializer="glorot_uniform", kernel_regularizer=l1(0.001)))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size

# def best_cnn_hw_nopoi_15000_chesctf(classes, number_of_samples):
#     # Best Convolutional Neural Network for CHES CTF dataset
#     # Number of points-of-interest: 15000
#     # Leakage model: HW
#     # POI interval: [0, 150000]
#     # Number of parameters: 5243209
#
#     batch_size = 100
#     seed = 91892
#     tf.random.set_seed(seed)
#     model = Sequential(name="best_cnn_hw_nopoi_chesctf_15000")
#     model.add(Conv1D(filters=8, kernel_size=10, strides=10, activation='relu', kernel_initializer='GlorotUniform', bias_initializer='Zeros',
#                      padding='valid', input_shape=(number_of_samples, 1)))
#     model.add(BatchNormalization())
#     model.add(AveragePooling1D(pool_size=2, strides=2))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu', kernel_initializer='GlorotUniform', bias_initializer='Zeros', kernel_regularizer=l1(l=0.0005)))
#     model.add(Dense(100, activation='relu', kernel_initializer='GlorotUniform', bias_initializer='Zeros', kernel_regularizer=l1(l=0.0005)))
#     model.add(Dense(classes, activation='softmax'))
#     model.summary()
#     optimizer = Adam(lr=0.001)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model, batch_size, seed
#
#
# def best_cnn_id_nopoi_15000_chesctf(classes, number_of_samples):
#     # Best Convolutional Neural Network for CHES CTF dataset
#     # Number of points-of-interest: 15000
#     # Leakage model: HW
#     # POI interval: [0, 150000]
#     # Number of parameters: 5243209
#
#     batch_size = 50
#     seed = 659276
#     tf.random.set_seed(seed)
#     model = Sequential(name="best_cnn_id_nopoi_chesctf_15000")
#     model.add(Conv1D(filters=8, kernel_size=10, strides=10, activation='relu', padding='valid', input_shape=(number_of_samples, 1)))
#     model.add(BatchNormalization())
#     model.add(AveragePooling1D(pool_size=2, strides=2))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu', kernel_regularizer=l1(l=0.0005)))
#     model.add(Dense(100, activation='relu', kernel_regularizer=l1(l=0.0005)))
#     model.add(Dense(classes, activation='softmax'))
#     model.summary()
#     optimizer = Adam(lr=0.001)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model, batch_size, seed
