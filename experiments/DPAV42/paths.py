from experiments.paths import *

directory_dataset = {
    "RPOI": f"{dataset_folder_dpav42}/DPAV42_rpoi",
    "OPOI": f"{dataset_folder_dpav42}/DPAV42_opoi",
    "NOPOI": f"{dataset_folder_dpav42}/DPAV42_nopoi"
}

directory_save_folder = {
    "RPOI": f"{results_folder_dpav42}/DPAV42_rpoi/random_search",
    "OPOI": f"{results_folder_dpav42}/DPAV42_opoi/random_search",
    "NOPOI": f"{results_folder_dpav42}/DPAV42_nopoi/random_search"
}

directory_save_folder_best_models = {
    "NOPOI": f"{results_folder_dpav42}/DPAV42_nopoi/best_models"
}


def dataset_name(feature_selection_type, npoi, window):
    dataset_name = {
        "RPOI": f"dpa_v42_{npoi}poi.h5",
        "OPOI": "dpa_v42_opoi.h5",
        "NOPOI": f"dpa_v42_nopoi_window_{window}.h5"
    }

    return dataset_name[feature_selection_type]


def dataset_name_desync(feature_selection_type, window=10):
    dataset_name = {
        "NOPOI": f"dpa_v42_nopoi_window_{window}_desync.h5"
    }

    return dataset_name[feature_selection_type]
