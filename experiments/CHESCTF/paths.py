from experiments.paths import *

directory_dataset = {
    "OPOI": f"{dataset_folder_chesctf}/CHESCTF_opoi",
    "NOPOI": f"{dataset_folder_chesctf}/CHESCTF_nopoi"
}

directory_save_folder = {
    "OPOI": f"{results_folder_chesctf}/CHESCTF_opoi/random_search",
    "NOPOI": f"{results_folder_chesctf}/CHESCTF_nopoi/random_search"
}

directory_save_folder_best_models = {
    "NOPOI": f"{results_folder_chesctf}/CHESCTF_nopoi/best_models"
}


def dataset_name(feature_selection_type, npoi, window):
    dataset_name = {
        "OPOI": "ches_ctf_opoi.h5",
        "NOPOI": f"ches_ctf_nopoi_window_{window}.h5"
    }

    return dataset_name[feature_selection_type]


def dataset_name_desync(feature_selection_type, window=10):
    dataset_name = {
        "NOPOI": f"ches_ctf_nopoi_window_{window}_desync.h5"
    }

    return dataset_name[feature_selection_type]
