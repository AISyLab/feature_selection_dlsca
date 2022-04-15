root_folder = "/tudelft.net/staff-umbrella/dlsca/Guilherme/ASCADf/"

directory_dataset = {
    "RPOI": f"{root_folder}/ASCAD_rpoi",
    "OPOI": f"{root_folder}/ASCAD_opoi",
    "SOPOI": f"{root_folder}/ASCAD_sppoi",
    "NOPOI": f"{root_folder}/ASCAD_nopoi"
}

directory_save_folder = {
    "RPOI": f"{root_folder}/ASCAD_rpoi/random_search",
    "OPOI": f"{root_folder}/ASCAD_opoi/random_search",
    "SOPOI": f"{root_folder}/ASCAD_sppoi/random_search",
    "NOPOI": f"{root_folder}/ASCAD_nopoi/random_search"
}

directory_save_folder_best_models = {
    "NOPOI": f"{root_folder}/ASCAD_nopoi/best_models"
}


def dataset_name(feature_selection_type, npoi, window=10):
    dataset_name = {
        "RPOI": f"ASCAD_{npoi}poi.h5",
        "OPOI": "ASCAD_opoi.h5",
        "SOPOI": "ASCAD_sopoi.h5",
        "NOPOI": f"ASCAD_nopoi_window_{window}.h5"
    }

    return dataset_name[feature_selection_type]


def dataset_name_desync(feature_selection_type, window=10):
    dataset_name = {
        "NOPOI": f"ASCAD_nopoi_window_{window}_desync.h5"
    }

    return dataset_name[feature_selection_type]
