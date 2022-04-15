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

from src.datasets.ReadASCADr import ReadASCADr
from src.datasets.dataset_parameters import *
from src.sca_metrics.sca_metrics import sca_metrics
from experiments.ASCADr.paths import *

if __name__ == "__main__":
    leakage_model = sys.argv[1]
    model_name = sys.argv[2]
    feature_selection_type = sys.argv[3]
    npoi = int(sys.argv[4])
    target_byte = int(sys.argv[5])
    window = int(sys.argv[6])
    desync = True if sys.argv[7] == "True" else False

    data_folder = directory_dataset[feature_selection_type]
    save_folder = directory_save_folder_best_models[feature_selection_type]
    if desync:
        filename = f"{data_folder}/{dataset_name_desync(feature_selection_type, window=window)}"
    else:
        filename = f"{data_folder}/{dataset_name(feature_selection_type, npoi, window=window)}"

    """ Parameters for the analysis """
    classes = 9 if leakage_model == "HW" else 256
    target_byte = 2
    epochs = 100
    ascadf_parameters = ascadf
    n_profiling = ascadf_parameters["n_profiling"]
    n_attack = ascadf_parameters["n_attack"]
    n_validation = ascadf_parameters["n_validation"]
    n_attack_ge = ascadf_parameters["n_attack_ge"]
    n_validation_ge = ascadf_parameters["n_validation_ge"]

    """ Create dataset for ASCADf """
    ascad_dataset = ReadASCADr(
        n_profiling,
        n_attack,
        n_validation,
        file_path=f"{filename}.h5",
        target_byte=target_byte,
        leakage_model=leakage_model,
        first_sample=0,
        number_of_samples=npoi,
        reshape_to_cnn=False if model_name == "mlp" else True,
    )

    start_time = time.time()

    """ Create random model """
    module_name = importlib.import_module(f"experiments.ASCADr.{feature_selection_type}.best_models")
    model_class = getattr(module_name, f"best_{model_name}_{leakage_model.lower()}_{feature_selection_type.lower()}_{npoi}_ascadr")
    model, batch_size = model_class()

    """ Train model """
    history = model.fit(
        x=ascad_dataset.x_profiling,
        y=ascad_dataset.y_profiling,
        batch_size=batch_size,
        verbose=2,
        epochs=100,
        shuffle=True,
        validation_data=(ascad_dataset.x_validation, ascad_dataset.y_validation),
        callbacks=[])

    """ Get DL metrics """
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    """ Compute GE, SR and NT for validation set """
    ge_validation, sr_validation, nt_validation = sca_metrics(
        model, ascad_dataset.x_validation, n_validation_ge, ascad_dataset.labels_key_hypothesis_validation, ascad_dataset.correct_key)

    print(f"GE validation: {ge_validation[n_validation_ge - 1]}")
    print(f"SR validation: {sr_validation[n_validation_ge - 1]}")
    print(f"Number of traces to reach GE = 1: {nt_validation}")

    """ Compute GE, SR and NT for attack set """
    ge_attack, sr_attack, nt_attack = sca_metrics(
        model, ascad_dataset.x_attack, n_attack_ge, ascad_dataset.labels_key_hypothesis_attack, ascad_dataset.correct_key)

    print(f"GE attack: {ge_attack[n_attack_ge - 1]}")
    print(f"SR attack: {sr_attack[n_attack_ge - 1]}")
    print(f"Number of traces to reach GE = 1: {nt_attack}")

    total_time = time.time() - start_time

    hp = None

    """ Create dictionary with results """
    npz_dict = {"npoi": npoi, "target_byte": target_byte, "n_profiling": n_profiling, "n_attack": n_attack,
                "n_validation": n_validation, "n_attack_ge": n_attack_ge, "n_validation_ge": n_validation_ge, "hp": hp,
                "ge_validation": ge_validation, "sr_validation": sr_validation, "nt_validation": nt_validation, "ge_attack": ge_attack,
                "sr_attack": sr_attack, "nt_attack": nt_attack, "accuracy": accuracy, "val_accuracy": val_accuracy, "loss": loss,
                "val_loss": val_loss, "elapsed_time": total_time, "params": model.count_params()}

    """ Save npz file with results """
    np.savez(f"{save_folder}/ascad-variable_{model_name}_{leakage_model}_{npoi}_{target_byte}.npz", npz_dict=npz_dict)