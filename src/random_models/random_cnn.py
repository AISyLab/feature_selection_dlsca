import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *
import numpy as np
from src.hyperparameters.hyperparameters import *
import importlib


def get_reg(hp):
    if hp["regularization"] == "l1":
        return l1(l=hp["l1"])
    elif hp["regularization"] == "l2":
        return l2(l=hp["l2"])
    else:
        return hp["dropout"]


def cnn_random(classes, number_of_samples, regularization=False):
    hp = get_hyperparemeters_cnn(regularization=regularization)

    tf_random_seed = np.random.randint(1048576)
    tf.random.set_seed(tf_random_seed)

    inputs = Input(shape=(number_of_samples, 1))

    x = None
    for layer_index in range(hp["conv_layers"]):
        x = Conv1D(kernel_size=hp["kernels"][layer_index], strides=hp["strides"][layer_index], filters=hp["filters"][layer_index],
                   activation=hp["activation"], padding="same")(inputs if layer_index == 0 else x)
        if hp["pooling_types"][layer_index] == "average":
            x = AveragePooling1D(pool_size=hp["pooling_sizes"][layer_index], strides=hp["pooling_strides"][layer_index], padding="same")(x)
        else:
            x = MaxPooling1D(pool_size=hp["pooling_sizes"][layer_index], strides=hp["pooling_strides"][layer_index], padding="same")(x)
        x = BatchNormalization()(x)
    x = Flatten()(x)

    for layer_index in range(hp["layers"]):
        if regularization and hp["regularization"] != "dropout":
            x = Dense(hp["neurons"], activation=hp["activation"], kernel_regularizer=get_reg(hp),
                      kernel_initializer=hp["kernel_initializer"], name='dense_{}'.format(layer_index))(x)
        else:
            x = Dense(hp["neurons"], activation=hp["activation"], kernel_initializer=hp["kernel_initializer"],
                      name='dense_{}'.format(layer_index))(x)
        if regularization and hp["regularization"] == "dropout":
            x = Dropout(get_reg(hp))(x)

    outputs = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, outputs, name='random_cnn')
    optimizer = get_optimizer(hp["optimizer"], hp["learning_rate"])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model, tf_random_seed, hp


def get_optimizer(optimizer, learning_rate):
    module_name = importlib.import_module("tensorflow.keras.optimizers")
    optimizer_class = getattr(module_name, optimizer)
    return optimizer_class(lr=learning_rate)
