import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *


def best_mlp_hw_nopoi_25000_ascadr(classes, number_of_samples):
    # Best multilayer perceptron for ASCAD variable key dataset
    # Number of points-of-interest: 25000
    # Leakage model: HW
    # POI interval: [0, 100000]
    # Number of parameters: 5243209

    batch_size = 700
    tf.random.set_seed(567322)
    model = Sequential(name='best_mlp_hw_nopoi_ascadr_25000')
    model.add(Dense(200, activation='selu', kernel_initializer='glorot_uniform', input_shape=(number_of_samples,)))
    model.add(Dense(200, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(200, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(200, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(200, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(200, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(200, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size


def best_mlp_id_nopoi_25000_ascadr(classes, number_of_samples):
    # Best multilayer perceptron for ASCAD variable key dataset
    # Number of points-of-interest: 25000
    # Leakage model: ID
    # POI interval: [0, 100000]
    # Number of parameters: 12628756

    batch_size = 100
    tf.random.set_seed(694875)
    model = Sequential(name='best_mlp_id_nopoi_ascadr_25000')
    model.add(Dense(500, activation='relu', kernel_initializer='he_uniform', input_shape=(number_of_samples,)))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size


def best_cnn_hw_nopoi_25000_ascadr(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCAD variable key dataset
    # Number of points-of-interest: 25000
    # Leakage model: HW
    # POI interval: [0, 100000]
    # Number of parameters: 369109

    batch_size = 400
    tf.random.set_seed(647729)
    model = Sequential(name='best_cnn_hw_nopoi_ascadr_25000')
    model.add(Conv1D(kernel_size=40, strides=20, filters=4, activation='selu', input_shape=(number_of_samples, 1), padding='same'))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=42, strides=21, filters=8, activation='selu', padding='same'))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=32, strides=16, filters=16, activation='selu', padding='same'))
    model.add(AveragePooling1D(pool_size=10, strides=10, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=50, strides=25, filters=32, activation='selu', padding='same'))
    model.add(AveragePooling1D(pool_size=8, strides=8, padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(400, activation='selu', kernel_initializer='random_uniform'))
    model.add(Dense(400, activation='selu', kernel_initializer='random_uniform'))
    model.add(Dense(400, activation='selu', kernel_initializer='random_uniform'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = RMSprop(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size


def best_cnn_id_nopoi_25000_ascadr(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCAD variable key dataset
    # Number of points-of-interest: 25000
    # Leakage model: ID
    # POI interval: [0, 100000]
    # Number of parameters: 721012

    batch_size = 900
    tf.random.set_seed(161399)
    model = Sequential(name='best_cnn_id_nopoi_ascadr_25000')
    model.add(Conv1D(kernel_size=34, strides=17, filters=4, activation='selu', input_shape=(number_of_samples, 1), padding='same'))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(200, activation='selu', kernel_initializer='random_uniform'))
    model.add(Dense(200, activation='selu', kernel_initializer='random_uniform'))
    model.add(Dense(200, activation='selu', kernel_initializer='random_uniform'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size
