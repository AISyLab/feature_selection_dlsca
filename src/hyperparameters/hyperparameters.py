import random


def get_hyperparameters_mlp(regularization=False, max_dense_layers=8):
    if regularization:
        return {
            "batch_size": random.randrange(100, 1100, 100),
            "layers": random.randrange(1, max_dense_layers + 1, 1),
            "neurons": random.choice([10, 20, 50, 100, 200, 300, 400, 500]),
            "activation": random.choice(["relu", "selu"]),
            "learning_rate": random.choice([0.005, 0.001, 0.0005, 0.0001]),
            "optimizer": random.choice(["Adam", "RMSprop"]),
            "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
            "regularization": random.choice(["l1", "l2", "dropout"]),
            "l1": random.choice([0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]),
            "l2": random.choice([0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]),
            "dropout": random.choice([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        }
    else:
        return {
            "batch_size": random.randrange(100, 1100, 100),
            "layers": random.choice([1, 2, 3, 4]),
            "neurons": random.choice([10, 20, 50, 100, 200, 300, 400, 500]),
            "activation": random.choice(["relu", "selu"]),
            "learning_rate": random.choice([0.005, 0.001, 0.0005, 0.0001]),
            "optimizer": random.choice(["Adam", "RMSprop"]),
            "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
            "regularization": random.choice(["none"])
        }


def get_hyperparemeters_cnn(regularization=False):
    hyperparameters = {}
    hyperparameters_mlp = get_hyperparameters_mlp(regularization=regularization, max_dense_layers=4)
    for key, value in hyperparameters_mlp.items():
        hyperparameters[key] = value

    conv_layers = random.choice([1, 2, 3, 4])
    kernels = []
    strides = []
    filters = []
    pooling_types = []
    pooling_sizes = []
    pooling_strides = []
    pooling_type = random.choice(["Average", "Max"])

    for conv_layer in range(1, conv_layers + 1):
        kernel = random.randrange(26, 52, 2)
        kernels.append(kernel)
        strides.append(int(kernel / 2))
        if conv_layer == 1:
            filters.append(random.choice([4, 8, 12, 16]))
        else:
            filters.append(filters[conv_layer - 2] * 2)
        pool_size = random.choice([2, 4, 6, 8, 10])
        pooling_sizes.append(pool_size)
        pooling_strides.append(pool_size)
        pooling_types.append(pooling_type)

    hyperparameters["conv_layers"] = conv_layers
    hyperparameters["kernels"] = kernels
    hyperparameters["strides"] = strides
    hyperparameters["filters"] = filters
    hyperparameters["pooling_sizes"] = pooling_sizes
    hyperparameters["pooling_strides"] = pooling_strides
    hyperparameters["pooling_types"] = pooling_types

    return hyperparameters
