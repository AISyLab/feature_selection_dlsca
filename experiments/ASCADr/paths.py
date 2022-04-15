from experiments.paths import *

directory_dataset = {
    "RPOI": f"{dataset_folder_ascadr}/ascad-variable_rpoi",
    "OPOI": f"{dataset_folder_ascadr}/ascad-variable_opoi",
    "NOPOI": f"{dataset_folder_ascadr}/ascad-variable_nopoi"
}

directory_save_folder = {
    "RPOI": f"{results_folder_ascadr}/ascad-variable_rpoi/random_search",
    "OPOI": f"{results_folder_ascadr}/ascad-variable_opoi/random_search",
    "NOPOI": f"{results_folder_ascadr}/ascad-variable_nopoi/random_search"
}

directory_save_folder_best_models = {
    "NOPOI": f"{results_folder_ascadr}ascad-variable_nopoi/best_models"
}


def dataset_name(feature_selection_type, npoi, window=20):
    dataset_name = {
        "RPOI": f"ascad-variable_{npoi}poi.h5",
        "OPOI": "ascad-variable_opoi.h5",
        "NOPOI": f"ascad-variable_nopoi_window_{window}.h5"
    }

    return dataset_name[feature_selection_type]


def dataset_name_desync(feature_selection_type, window=10):
    dataset_name = {
        "NOPOI": f"ascad-variable_nopoi_window_{window}_desync.h5"
    }

    return dataset_name[feature_selection_type]
