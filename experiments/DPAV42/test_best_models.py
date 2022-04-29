import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

import os
import sys
import time
import glob
import numpy as np

sys.path.append('/home/nfs/gperin/feature_selection_paper')

os.environ["OMP_NUM_THREADS"] = '2'  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '2'  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '2'  # export MKL_NUM_THREADS=6

import importlib

from src.datasets.ReadDPAV42 import ReadDPAV42
from src.datasets.dataset_parameters import *
from src.sca_metrics.sca_metrics import sca_metrics
from experiments.paths import *


def dataset_name(fs_type, num_poi, resampling_window):
    dataset_name = {
        "RPOI": f"dpa_v42_{num_poi}poi.h5",
        "OPOI": "dpa_v42_opoi.h5",
        "NOPOI": f"dpa_v42_nopoi_window_{resampling_window}.h5",
        "NOPOI_DESYNC": f"dpa_v42_nopoi_window_{resampling_window}_desync.h5"
    }

    return dataset_name[fs_type]


if __name__ == "__main__":
    leakage_model = sys.argv[1]
    model_name = sys.argv[2]
    feature_selection_type = sys.argv[3]
    npoi = int(sys.argv[4])
    target_byte = int(sys.argv[5])
    window = int(sys.argv[6])

    if feature_selection_type == "RPOI":
        dataset_folder = dataset_folder_dpav42_rpoi
        save_folder = results_folder_dpav42_rpoi
    elif feature_selection_type == "OPOI":
        dataset_folder = dataset_folder_dpav42_opoi
        save_folder = results_folder_dpav42_opoi
    elif feature_selection_type == "NOPOI":
        dataset_folder = dataset_folder_dpav42_nopoi
        save_folder = results_folder_dpav42_nopoi
    elif feature_selection_type == "NOPOI_DESYNC":
        dataset_folder = dataset_folder_dpav42_nopoi_desync
        save_folder = results_folder_dpav42_nopoi_desync
    else:
        dataset_folder = None
        save_folder = None
        print("ERROR: Feature selection type not found.")
        exit()

    filename = f"{dataset_folder}/{dataset_name(feature_selection_type, npoi, resampling_window=window)}"

    """ Parameters for the analysis """
    classes = 9 if leakage_model == "HW" else 256
    epochs = 100
    dpav42_parameters = dpav42
    n_profiling = dpav42_parameters["n_profiling"]
    n_attack = dpav42_parameters["n_attack"]
    n_validation = dpav42_parameters["n_validation"]
    n_attack_ge = dpav42_parameters["n_attack_ge"]
    n_validation_ge = dpav42_parameters["n_validation_ge"]

    """ Create dataset for DPAV42 """
    dpav42_dataset = ReadDPAV42(
        n_profiling,
        n_attack,
        n_validation,
        file_path=f"{filename}",
        target_byte=target_byte,
        leakage_model=leakage_model,
        first_sample=0,
        number_of_samples=npoi,
        reshape_to_cnn=False if model_name == "mlp" else True,
    )

    """ Create random model """
    module_name = importlib.import_module(f"experiments.DPAV42.{feature_selection_type}.best_models")
    model_class = getattr(module_name, f"best_{model_name}_{leakage_model.lower()}_{feature_selection_type.lower()}_{npoi}_dpav42")
    model, batch_size = model_class(classes, npoi)

    """ Train model """
    history = model.fit(
        x=dpav42_dataset.x_profiling,
        y=dpav42_dataset.y_profiling,
        batch_size=batch_size,
        verbose=2,
        epochs=100,
        shuffle=True,
        validation_data=(dpav42_dataset.x_validation, dpav42_dataset.y_validation),
        callbacks=[])

    """ Get DL metrics """
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    """ Compute GE, SR and NT for validation set """
    ge_validation, sr_validation, nt_validation = sca_metrics(
        model, dpav42_dataset.x_validation, n_validation_ge, dpav42_dataset.labels_key_hypothesis_validation,
        dpav42_dataset.correct_key_validation)

    print(f"GE validation: {ge_validation[n_validation_ge - 1]}")
    print(f"SR validation: {sr_validation[n_validation_ge - 1]}")
    print(f"Number of traces to reach GE = 1: {nt_validation}")

    """ Compute GE, SR and NT for attack set """
    ge_attack, sr_attack, nt_attack = sca_metrics(
        model, dpav42_dataset.x_attack, n_attack_ge, dpav42_dataset.labels_key_hypothesis_attack, dpav42_dataset.correct_key_attack)

    print(f"GE attack: {ge_attack[n_attack_ge - 1]}")
    print(f"SR attack: {sr_attack[n_attack_ge - 1]}")
    print(f"Number of traces to reach GE = 1: {nt_attack}")
