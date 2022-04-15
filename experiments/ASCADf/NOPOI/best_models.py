import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *


def best_mlp_hw_nopoi_10000_ascadf(classes, number_of_samples):
    # Best multilayer perceptron for ASCAD fixed key dataset
    # Number of points-of-interest: 10000
    # Leakage model: HW
    # POI interval: [0, 100000]
    # Number of parameters: 2203009

    batch_size = 300
    tf.random.set_seed(139352)
    model = Sequential(name='best_mlp_hw_nopoi_ascadv_10000')
    model.add(Dense(200, activation='selu', kernel_regularizer=l1(l=0.0001), kernel_initializer="he_uniform", input_shape=(number_of_samples,)))
    model.add(Dense(200, activation='selu', kernel_regularizer=l1(l=0.0001), kernel_initializer="he_uniform"))
    model.add(Dense(200, activation='selu', kernel_regularizer=l1(l=0.0001), kernel_initializer="he_uniform"))
    model.add(Dense(200, activation='selu', kernel_regularizer=l1(l=0.0001), kernel_initializer="he_uniform"))
    model.add(Dense(200, activation='selu', kernel_regularizer=l1(l=0.0001), kernel_initializer="he_uniform"))
    model.add(Dense(200, activation='selu', kernel_regularizer=l1(l=0.0001), kernel_initializer="he_uniform"))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size


def best_mlp_id_nopoi_10000_ascadf(classes, number_of_samples):
    # Best multilayer perceptron for ASCAD fixed key dataset
    # Number of points-of-interest: 10000
    # Leakage model: ID
    # POI interval: [0, 100000]
    # Number of parameters: 5379256

    batch_size = 100
    tf.random.set_seed(24638)
    model = Sequential(name='best_mlp_id_nopoi_ascadv_10000')
    model.add(Dense(500, activation='selu', kernel_regularizer=l2(l=0.0005), kernel_initializer="random_uniform", input_shape=(number_of_samples,)))
    model.add(Dense(500, activation='selu', kernel_regularizer=l2(l=0.0005), kernel_initializer="random_uniform"))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = RMSprop(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size


def best_cnn_hw_nopoi_10000_ascadf(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCAD fixed key dataset
    # Number of points-of-interest: 10000
    # Leakage model: HW
    # POI interval: [0, 100000]
    # Number of parameters: 545693

    batch_size = 300
    tf.random.set_seed(844736)
    model = Sequential(name='best_cnn_hw_nopoi_ascadv_10000')
    model.add(Conv1D(kernel_size=38, strides=19, filters=4, activation='selu', input_shape=(number_of_samples, 1), padding='same'))
    model.add(AveragePooling1D(pool_size=4, strides=4, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=46, strides=23, filters=8, activation='selu', padding='same'))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=26, strides=13, filters=16, activation='selu', padding='same'))
    model.add(AveragePooling1D(pool_size=6, strides=6, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=36, strides=18, filters=32, activation='selu', padding='same'))
    model.add(AveragePooling1D(pool_size=6, strides=6, padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(500, activation='selu', kernel_initializer="he_uniform"))
    model.add(Dense(500, activation='selu', kernel_initializer="he_uniform"))
    model.add(Dense(500, activation='selu', kernel_initializer="he_uniform"))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size


def best_cnn_id_nopoi_10000_ascadf(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCAD fixed key dataset
    # Number of points-of-interest: 10000
    # Leakage model: ID
    # POI interval: [0, 100000]
    # Number of parameters: 439348

    batch_size = 800
    tf.random.set_seed(1012931)
    model = Sequential(name='best_cnn_id_nopoi_ascadv_10000')
    model.add(Conv1D(kernel_size=44, strides=47, filters=12, activation='selu', input_shape=(number_of_samples, 1), padding='same'))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=18, strides=46, filters=24, activation='selu', padding='same'))
    model.add(AveragePooling1D(pool_size=6, strides=6, padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(400, activation='selu', kernel_initializer="random_uniform"))
    model.add(Dense(400, activation='selu', kernel_initializer="random_uniform"))
    model.add(Dense(400, activation='selu', kernel_initializer="random_uniform"))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size
