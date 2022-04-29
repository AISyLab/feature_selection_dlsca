import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *


def best_mlp_hw_opoi_700_ascadf(classes, number_of_samples):
    # Best multilayer perceptron for ASCAD fixed key dataset
    # Number of points-of-interest: 700
    # Leakage model: HW
    # POI interval: [45400, 46100]
    # Number of parameters: 16309

    batch_size = 900
    tf.random.set_seed(978883)
    model = Sequential(name='best_mlp_hw_opoi_ascadf_700')
    model.add(Dense(20, activation='relu', kernel_initializer='random_uniform', input_shape=(number_of_samples,)))
    model.add(Dense(20, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(20, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(20, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(20, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(20, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size


def best_mlp_id_opoi_700_ascadf(classes, number_of_samples):
    # Best multilayer perceptron for ASCAD fixed key dataset
    # Number of points-of-interest: 700
    # Leakage model: ID
    # POI interval: [45400, 46100]
    # Number of parameters: 10266

    batch_size = 1000
    tf.random.set_seed(114621)
    model = Sequential(name='best_mlp_id_opoi_ascadf_700')
    model.add(Dense(10, activation='selu', kernel_initializer='random_uniform', input_shape=(number_of_samples,)))
    model.add(Dense(10, activation='selu', kernel_initializer='random_uniform'))
    model.add(Dense(10, activation='selu', kernel_initializer='random_uniform'))
    model.add(Dense(10, activation='selu', kernel_initializer='random_uniform'))
    model.add(Dense(10, activation='selu', kernel_initializer='random_uniform'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size


def best_cnn_hw_opoi_700_ascadf(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCAD fixed key dataset
    # Number of points-of-interest: 700
    # Leakage model: HW
    # POI interval: [45400, 46100]
    # Number of parameters: 594305

    batch_size = 900
    tf.random.set_seed(124124)
    model = Sequential(name='best_cnn_hw_opoi_ascadf_700')
    model.add(Conv1D(kernel_size=26, strides=13, filters=8, activation='relu', input_shape=(number_of_samples, 1), padding='same'))
    model.add(MaxPool1D(pool_size=4, strides=4, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=38, strides=19, filters=16, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=4, strides=4, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=32, strides=16, filters=32, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=30, strides=15, filters=64, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=8, strides=8, padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(400, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(400, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(400, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(400, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, batch_size


def best_cnn_id_opoi_700_ascadf(classes, number_of_samples):
    # Best Convolutional Neural Network for ASCADf key dataset
    # Number of points-of-interest: 700
    # Leakage model: ID
    # Number of parameters: 7776

    batch_size = 100
    tf.random.set_seed(610875)
    model = Sequential(name="best_cnn_id_rpoi_700_ascadf")
    model.add(Conv1D(kernel_size=26, strides=13, filters=8, activation="selu", input_shape=(number_of_samples, 1), padding="same"))
    model.add(MaxPool1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(kernel_size=34, strides=17, filters=16, activation="selu", padding="same"))
    model.add(MaxPool1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(10, activation="selu", kernel_initializer="random_uniform"))
    model.add(Dense(10, activation="selu", kernel_initializer="random_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = RMSprop(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size
