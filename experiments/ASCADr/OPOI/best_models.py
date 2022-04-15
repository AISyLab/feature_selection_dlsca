import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras import *


def best_mlp_hw_opoi_1400_ascadr(classes, number_of_samples):
    # Best multilayer perceptron for ASCAD variable key dataset
    # Number of points-of-interest: 1400
    # Leakage model: HW
    # POI interval: [80945, 82345]
    # Number of parameters: 31149

    batch_size = 400
    tf.random.set_seed(51262)
    model = Sequential(name='best_mlp_hw_opoi_ascadv_1400')
    model.add(Dense(20, activation='selu', kernel_initializer='glorot_uniform', input_shape=(number_of_samples,)))
    model.add(Dense(20, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(20, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(20, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(20, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(20, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(20, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(20, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = RMSprop(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size


def best_mlp_id_opoi_1400_ascadr(classes, number_of_samples):
    # Best multilayer perceptron for ASCAD variable key dataset
    # Number of points-of-interest: 1400
    # Leakage model: ID
    # POI interval: [80945, 82345]
    # Number of parameters: 34236

    batch_size = 100
    tf.random.set_seed(83545)
    model = Sequential(name='best_mlp_id_opoi_ascadv_1400')
    model.add(Dense(20, activation='selu', kernel_initializer='random_uniform', input_shape=(number_of_samples,)))
    model.add(Dense(20, activation='selu', kernel_initializer='random_uniform'))
    model.add(Dense(20, activation='selu', kernel_initializer='random_uniform'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size


def best_cnn_hw_opoi_1400_ascadr(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCAD variable key dataset
    # Number of points-of-interest: 1400
    # Leakage model: HW
    # POI interval: [80945, 82345]
    # Number of parameters: 270953

    batch_size = 500
    tf.random.set_seed(908232)
    model = Sequential(name='best_cnn_hw_opoi_ascadv_1400')
    model.add(Conv1D(kernel_size=32, strides=16, filters=16, activation='selu', input_shape=(number_of_samples, 1), padding='same'))
    model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=28, strides=14, filters=32, activation='selu', padding='same'))
    model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=32, strides=16, filters=64, activation='selu', padding='same'))
    model.add(MaxPool1D(pool_size=6, strides=6, padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(400, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(400, activation='selu', kernel_initializer='glorot_uniform'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = RMSprop(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size


def best_cnn_id_opoi_1400_ascadr(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCAD variable key dataset
    # Number of points-of-interest: 1400
    # Leakage model: ID
    # POI interval: [80945, 82345]
    # Number of parameters: 87632

    batch_size = 300
    tf.random.set_seed(511628)
    model = Sequential(name='best_cnn_id_opoi_ascadv_1400')
    model.add(Conv1D(kernel_size=46, strides=23, filters=8, activation='selu', input_shape=(number_of_samples, 1), padding='same'))
    model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=50, strides=25, filters=16, activation='selu', padding='same'))
    model.add(MaxPool1D(pool_size=6, strides=6, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=44, strides=22, filters=32, activation='selu', padding='same'))
    model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(200, activation='selu', kernel_initializer='he_uniform'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size
