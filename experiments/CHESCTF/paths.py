root_folder = "/tudelft.net/staff-umbrella/dlsca/Guilherme/CHESCTF"

directory_dataset = {
    "RPOI": f"{root_folder}/CHESCTF_rpoi",
    "OPOI": f"{root_folder}/CHESCTF_opoi",
    "SOPOI": f"{root_folder}/CHESCTF_sopoi",
    "NOPOI": f"{root_folder}/CHESCTF_nopoi"
}

directory_save_folder = {
    "RPOI": f"{root_folder}/CHESCTF_rpoi/random_search",
    "OPOI": f"{root_folder}/CHESCTF_opoi/random_search",
    "SOPOI": f"{root_folder}/CHESCTF_sopoi/random_search",
    "NOPOI": f"{root_folder}/CHESCTF_nopoi/random_search"
}

directory_save_folder_best_models = {
    "NOPOI": f"{root_folder}/CHESCTF_nopoi/best_models"
}


def dataset_name(feature_selection_type, npoi, window):
    dataset_name = {
        "RPOI": f"ches_ctf_{npoi}poi.h5",
        "OPOI": "ches_ctf_opoi.h5",
        "SOPOI": "ches_ctf_sopoi.h5",
        "NOPOI": f"ches_ctf_nopoi_window_{window}.h5"
    }

    return dataset_name[feature_selection_type]


def dataset_name_desync(feature_selection_type, window=10):
    dataset_name = {
        "NOPOI": f"ches_ctf_nopoi_window_{window}_desync.h5"
    }

    return dataset_name[feature_selection_type]
