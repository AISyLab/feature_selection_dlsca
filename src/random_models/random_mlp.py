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


def mlp_random(classes, number_of_samples, regularization=False, hp=None):
    hp = get_hyperparameters_mlp(regularization=regularization) if hp is None else hp

    tf_random_seed = np.random.randint(1048576)
    tf.random.set_seed(tf_random_seed)

    inputs = Input(shape=number_of_samples)

    x = None
    for layer_index in range(hp["layers"]):
        if regularization and hp["regularization"] != "dropout":
            x = Dense(hp["neurons"], activation=hp["activation"], kernel_regularizer=get_reg(hp),
                      kernel_initializer=hp["kernel_initializer"],
                      name='dense_{}'.format(layer_index))(inputs if layer_index == 0 else x)
        else:
            x = Dense(hp["neurons"], activation=hp["activation"], kernel_initializer=hp["kernel_initializer"],
                      name='dense_{}'.format(layer_index))(inputs if layer_index == 0 else x)
        if regularization and hp["regularization"] == "dropout":
            x = Dropout(get_reg(hp))(x)

    outputs = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, outputs, name='random_mlp')
    optimizer = get_optimizer(hp["optimizer"], hp["learning_rate"])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model, tf_random_seed, hp


def get_optimizer(optimizer, learning_rate):
    module_name = importlib.import_module("tensorflow.keras.optimizers")
    optimizer_class = getattr(module_name, optimizer)
    return optimizer_class(lr=learning_rate)
