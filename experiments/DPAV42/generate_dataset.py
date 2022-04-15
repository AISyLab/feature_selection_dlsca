import numpy as np
from scalib.metrics import SNR
import matplotlib.pyplot as plt
import h5py
from numba import njit
from paths import *
from tqdm import tqdm
import random

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


@njit
def winres(trace, window=20, overlap=0.5):
    trace_winres = []
    step = int(window * overlap)
    max = len(trace)
    for i in range(0, max, step):
        trace_winres.append(np.mean(trace[i:i + window]))
    return np.array(trace_winres)


def compute_snr(feature_selection_type, window=10):
    in_file = h5py.File(f'{directory_dataset[feature_selection_type]}/dpa_v42_{feature_selection_type.lower()}.h5', "r")
    profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
    profiling_keys = in_file['Profiling_traces/metadata']['key']
    profiling_plaintexts = in_file['Profiling_traces/metadata']['plaintext']
    profiling_masks = in_file['Profiling_traces/metadata']['masks']

    mask = [3, 12, 53, 58, 80, 95, 102, 105, 150, 153, 160, 175, 197, 202, 243, 252]
    mask_substitution = np.zeros(256)
    m = 0
    for i in range(256):
        if i == mask[m]:
            mask_substitution[i] = m
            m += 1
            if m == 16:
                break

    mask_shares = [[int(r[0])] for r in zip(np.asarray(profiling_masks[:, 0]))]
    mask_shares = mask_substitution[mask_shares]

    snr = SNR(np=1, ns=len(profiling_samples[0]), nc=16)
    snr.fit_u(np.array(profiling_samples, dtype=np.int16), x=np.array(mask_shares, dtype=np.uint16))
    snr_val_r2 = snr.get_snr()

    mask_shares = [[AES_Sbox[int(p) ^ int(k)] ^ int(r)] for p, k, r in
                   zip(np.asarray(profiling_plaintexts[:, 0]), np.asarray(profiling_keys[:, 0]), np.asarray(profiling_masks[:, 0]))]
    snr = SNR(np=1, ns=len(profiling_samples[0]), nc=256)
    snr.fit_u(np.array(profiling_samples, dtype=np.int16), x=np.array(mask_shares, dtype=np.uint16))
    snr_val_sm = snr.get_snr()

    figure = plt.gcf()
    figure.set_size_inches(12, 3)
    plt.plot(snr_val_r2[0], linewidth=1, zorder=1, label="$r_{0}$")
    plt.plot(snr_val_sm[0], linewidth=1, zorder=1, label="$S(p_{0}\oplus k_{0}) \oplus r_{0}$")
    plt.xlabel("Samples", fontsize=12)
    plt.ylabel("SNR", fontsize=12)
    plt.xlim([0, len(profiling_samples[0])])
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{directory_dataset[feature_selection_type]}/dpa_v42_{feature_selection_type.lower()}_snr_15000.png", dpi=100)
    plt.close()


def compute_snr_nopoi(window):
    in_file = h5py.File(f'{directory_dataset["NOPOI"]}/dpa_v42_nopoi_window_{window}.h5', "r")
    profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
    profiling_keys = in_file['Profiling_traces/metadata']['key']
    profiling_plaintexts = in_file['Profiling_traces/metadata']['plaintext']
    profiling_masks = in_file['Profiling_traces/metadata']['masks']

    mask = [3, 12, 53, 58, 80, 95, 102, 105, 150, 153, 160, 175, 197, 202, 243, 252]
    mask_substitution = np.zeros(256)
    m = 0
    for i in range(256):
        if i == mask[m]:
            mask_substitution[i] = m
            m += 1
            if m == 16:
                break

    mask_shares = [[int(r[0])] for r in zip(np.asarray(profiling_masks[:, 0]))]
    mask_shares = mask_substitution[mask_shares]

    snr = SNR(np=1, ns=len(profiling_samples[0]), nc=16)
    snr.fit_u(np.array(profiling_samples, dtype=np.int16), x=np.array(mask_shares, dtype=np.uint16))
    snr_val_r2 = snr.get_snr()

    mask_shares = [[AES_Sbox[int(p) ^ int(k)] ^ int(r)] for p, k, r in
                   zip(np.asarray(profiling_plaintexts[:, 0]), np.asarray(profiling_keys[:, 0]), np.asarray(profiling_masks[:, 0]))]
    snr = SNR(np=1, ns=len(profiling_samples[0]), nc=256)
    snr.fit_u(np.array(profiling_samples, dtype=np.int16), x=np.array(mask_shares, dtype=np.uint16))
    snr_val_sm = snr.get_snr()

    figure = plt.gcf()
    figure.set_size_inches(12, 3)
    plt.plot(snr_val_r2[0], linewidth=1, zorder=1, label="$r_{0}$")
    plt.plot(snr_val_sm[0], linewidth=1, zorder=1, label="$S(p_{0}\oplus k_{0}) \oplus r_{0}$")
    plt.xlabel("Samples", fontsize=12)
    plt.ylabel("SNR", fontsize=12)
    plt.xlim([0, len(profiling_samples[0])])
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{directory_dataset['NOPOI']}/dpa_v42_nopoi_snr_window_{window}.png", dpi=100)
    plt.close()


def compute_snr_rpoi(n_poi, leakage_model):
    if leakage_model == "ID":
        in_file = h5py.File(f'{directory_dataset["RPOI"]}/dpa_v42_{n_poi}poi.h5', 'r')
    else:
        in_file = h5py.File(f'{directory_dataset["RPOI"]}/dpa_v42_{n_poi}poi_hw.h5', 'r')

    profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
    profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
    profiling_key = in_file['Profiling_traces/metadata']['key']
    profiling_masks = in_file['Profiling_traces/metadata']['masks']

    mask = [3, 12, 53, 58, 80, 95, 102, 105, 150, 153, 160, 175, 197, 202, 243, 252]
    mask_substitution = np.zeros(256)
    m = 0
    for i in range(256):
        if i == mask[m]:
            mask_substitution[i] = m
            m += 1
            if m == 16:
                break

    mask_shares = [[int(r[0])] for r in zip(np.asarray(profiling_masks[:, 0]))]
    mask_shares = mask_substitution[mask_shares]

    snr = SNR(np=1, ns=len(profiling_samples[0]), nc=16)
    snr.fit_u(np.array(profiling_samples, dtype=np.int16), x=np.array(mask_shares, dtype=np.uint16))
    snr_val_r2 = snr.get_snr()

    mask_shares = [[bin(AES_Sbox[int(p) ^ int(k)] ^ int(r)).count("1")] if leakage_model == "HW" else [AES_Sbox[int(p) ^ int(k)] ^ int(r)]
                   for p, k, r in
                   zip(np.asarray(profiling_plaintext[:, 0]), np.asarray(profiling_key[:, 0]), np.asarray(profiling_masks[:, 0]))]
    snr = SNR(np=1, ns=len(profiling_samples[0]), nc=256 if leakage_model == "ID" else 9)
    snr.fit_u(np.array(profiling_samples, dtype=np.int16), x=np.array(mask_shares, dtype=np.uint16))
    snr_val_sm = snr.get_snr()

    figure = plt.gcf()
    figure.set_size_inches(12, 3)
    plt.plot(snr_val_r2[0], linewidth=1, zorder=1, label="$r_{2}$")
    plt.plot(snr_val_sm[0], linewidth=1, zorder=1, label="$S(p_{2}\oplus k_{2}) \oplus r_{2}$")
    plt.xlabel("Samples", fontsize=12)
    plt.ylabel("SNR", fontsize=12)
    plt.xlim([0, n_poi])
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if leakage_model == "ID":
        plt.savefig(f'{directory_dataset["RPOI"]}/dpa_v42_snr_{n_poi}poi.png', dpi=100)
    else:
        plt.savefig(f'{directory_dataset["RPOI"]}/dpa_v42_snr_{n_poi}poi_hw.png', dpi=100)
    plt.close()


def generate_rpoi(gaussian_noise=None):
    n_profiling = 70000
    n_attack = 10000

    for leakage_model in ["HW", "ID"]:

        print("Opening dataset")

        trace_folder = "/"
        in_file = h5py.File(f'{trace_folder}dpa_v42_full.h5', 'r')
        profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float16)
        attack_samples = np.array(in_file['Attack_traces/traces'], dtype=np.float16)
        profiling_keys = in_file['Profiling_traces/metadata']['key']
        attack_keys = in_file['Attack_traces/metadata']['key']
        profiling_plaintexts = in_file['Profiling_traces/metadata']['plaintext']
        attack_plaintexts = in_file['Attack_traces/metadata']['plaintext']
        profiling_ciphertexts = in_file['Profiling_traces/metadata']['ciphertext']
        attack_ciphertexts = in_file['Attack_traces/metadata']['ciphertext']
        profiling_masks = in_file['Profiling_traces/metadata']['masks']
        attack_masks = in_file['Attack_traces/metadata']['masks']

        if gaussian_noise is not None:
            print("Adding Gaussian Noise")
            for i in tqdm(range(len(profiling_samples))):
                noise = np.random.normal(0, gaussian_noise, size=np.shape(profiling_samples.shape[1]))
                profiling_samples[i] = np.add(np.array(profiling_samples[i]), noise, dtype=np.float16)

            for i in tqdm(range(len(attack_samples))):
                noise = np.random.normal(0, gaussian_noise, size=np.shape(attack_samples.shape[1]))
                attack_samples[i] = np.add(np.array(attack_samples[i]), noise, dtype=np.float16)

        print("Computing SNR")
        mask = [3, 12, 53, 58, 80, 95, 102, 105, 150, 153, 160, 175, 197, 202, 243, 252]
        mask_substitution = np.zeros(256)
        m = 0
        for i in range(256):
            if i == mask[m]:
                mask_substitution[i] = m
                m += 1
                if m == 16:
                    break

        ns = len(profiling_samples[0])
        mask_shares = [[int(r[0])] for r in zip(np.asarray(profiling_masks[:, 0]))]
        mask_shares = mask_substitution[mask_shares]

        snr = SNR(np=1, ns=ns, nc=16)
        snr.fit_u(np.array(profiling_samples, dtype=np.int16), x=np.array(mask_shares, dtype=np.uint16))
        snr_val_r2 = snr.get_snr()

        mask_shares = [
            [bin(AES_Sbox[int(p) ^ int(k)] ^ int(r)).count("1")] if leakage_model == "HW" else [AES_Sbox[int(p) ^ int(k)] ^ int(r)]
            for p, k, r in
            zip(np.asarray(profiling_plaintexts[:, 0]), np.asarray(profiling_keys[:, 0]), np.asarray(profiling_masks[:, 0]))]
        snr = SNR(np=1, ns=ns, nc=256 if leakage_model == "ID" else 9)
        snr.fit_u(np.array(profiling_samples, dtype=np.int16), x=np.array(mask_shares, dtype=np.uint16))
        snr_val_sm = snr.get_snr()

        """ log scale """
        poi_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

        for n_poi in poi_list:
            """ sort POIs from masked sbox output """
            ind_snr_masks_poi_sm = np.argsort(snr_val_sm[0])[::-1][:int(n_poi / 2)]
            ind_snr_masks_poi_sm_sorted = np.sort(ind_snr_masks_poi_sm)

            """ sort POIs from mask share """
            ind_snr_masks_poi_r2 = np.argsort(snr_val_r2[0])[::-1][:int(n_poi / 2)]
            ind_snr_masks_poi_r2_sorted = np.sort(ind_snr_masks_poi_r2)

            snr_masks_poi = np.concatenate((ind_snr_masks_poi_sm_sorted, ind_snr_masks_poi_r2_sorted), axis=0)
            snr_masks_poi = np.sort(snr_masks_poi)

            attack_samples_poi = attack_samples[:, snr_masks_poi]
            profiling_samples_poi = profiling_samples[:, snr_masks_poi]

            profiling_index = [n for n in range(n_profiling)]
            attack_index = [n for n in range(n_attack)]

            if leakage_model == "ID":
                out_file = h5py.File(f'{directory_dataset["RPOI"]}/dpa_v42_{n_poi}poi.h5', 'w')
            else:
                out_file = h5py.File(f'{directory_dataset["RPOI"]}/dpa_v42_{n_poi}poi_hw.h5', 'w')

            profiling_traces_group = out_file.create_group("Profiling_traces")
            attack_traces_group = out_file.create_group("Attack_traces")

            profiling_traces_group.create_dataset(name="traces", data=profiling_samples_poi, dtype=profiling_samples_poi.dtype)
            attack_traces_group.create_dataset(name="traces", data=attack_samples_poi, dtype=attack_samples_poi.dtype)

            metadata_type_profiling = np.dtype([("plaintext", profiling_plaintexts.dtype, (len(profiling_plaintexts[0]),)),
                                                ("ciphertext", profiling_ciphertexts.dtype, (len(profiling_ciphertexts[0]),)),
                                                ("key", profiling_keys.dtype, (len(profiling_keys[0]),)),
                                                ("masks", profiling_masks.dtype, (len(profiling_masks[0]),))
                                                ])
            metadata_type_attack = np.dtype([("plaintext", attack_plaintexts.dtype, (len(attack_plaintexts[0]),)),
                                             ("ciphertext", attack_ciphertexts.dtype, (len(attack_ciphertexts[0]),)),
                                             ("key", attack_keys.dtype, (len(attack_keys[0]),)),
                                             ("masks", attack_masks.dtype, (len(attack_masks[0]),))
                                             ])

            profiling_metadata = np.array(
                [(profiling_plaintexts[n], profiling_ciphertexts[n], profiling_keys[n], profiling_masks[n]) for n in
                 profiling_index], dtype=metadata_type_profiling)
            profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

            attack_metadata = np.array([(attack_plaintexts[n], attack_ciphertexts[n], attack_keys[n], attack_masks[n]) for n in
                                        attack_index], dtype=metadata_type_attack)
            attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

            out_file.flush()
            out_file.close()

            print(n_poi)

            compute_snr_rpoi(n_poi, leakage_model)


def generate_opoi():
    in_file = h5py.File(f'{directory_dataset["NOPOI"]}/dpa_v42_nopoi_window_20.h5', "r")
    profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float16)
    attack_samples = np.array(in_file['Attack_traces/traces'], dtype=np.float16)
    profiling_keys = in_file['Profiling_traces/metadata']['key']
    attack_keys = in_file['Attack_traces/metadata']['key']
    profiling_plaintexts = in_file['Profiling_traces/metadata']['plaintext']
    attack_plaintexts = in_file['Attack_traces/metadata']['plaintext']
    profiling_ciphertexts = in_file['Profiling_traces/metadata']['ciphertext']
    attack_ciphertexts = in_file['Attack_traces/metadata']['ciphertext']
    profiling_masks = in_file['Profiling_traces/metadata']['masks']
    attack_masks = in_file['Attack_traces/metadata']['masks']

    out_file = h5py.File(f'{directory_dataset["OPOI"]}/dpa_v42_opoi.h5', 'w')

    n_profiling = 70000
    n_attack = 10000

    profiling_samples_opoi = np.zeros((len(profiling_samples), 800))
    attack_samples_opoi = np.zeros((len(attack_samples), 800))

    for i in range(len(profiling_samples)):
        profiling_samples_opoi[i][:400] = profiling_samples[i][17000:17400]
        profiling_samples_opoi[i][400:] = profiling_samples[i][20600:21000]

    for i in range(len(attack_samples)):
        attack_samples_opoi[i][:400] = attack_samples[i][17000:17400]
        attack_samples_opoi[i][400:] = attack_samples[i][20600:21000]

    profiling_index = [n for n in range(n_profiling)]
    attack_index = [n for n in range(n_attack)]

    profiling_traces_group = out_file.create_group("Profiling_traces")
    attack_traces_group = out_file.create_group("Attack_traces")

    profiling_traces_group.create_dataset(name="traces", data=profiling_samples_opoi, dtype=profiling_samples_opoi.dtype)
    attack_traces_group.create_dataset(name="traces", data=attack_samples_opoi, dtype=attack_samples_opoi.dtype)

    metadata_type_profiling = np.dtype([("plaintext", profiling_plaintexts.dtype, (len(profiling_plaintexts[0]),)),
                                        ("ciphertext", profiling_ciphertexts.dtype, (len(profiling_ciphertexts[0]),)),
                                        ("key", profiling_keys.dtype, (len(profiling_keys[0]),)),
                                        ("masks", profiling_masks.dtype, (len(profiling_masks[0]),))])
    metadata_type_attack = np.dtype([("plaintext", attack_plaintexts.dtype, (len(attack_plaintexts[0]),)),
                                     ("ciphertext", attack_ciphertexts.dtype, (len(attack_ciphertexts[0]),)),
                                     ("key", attack_keys.dtype, (len(attack_keys[0]),)),
                                     ("masks", attack_masks.dtype, (len(attack_masks[0]),))])

    profiling_metadata = np.array([(profiling_plaintexts[n], profiling_ciphertexts[n], profiling_keys[n], profiling_masks[n]) for n, k in
                                   zip(profiling_index, range(n_profiling))], dtype=metadata_type_profiling)
    profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

    attack_metadata = np.array([(attack_plaintexts[n], attack_ciphertexts[n], attack_keys[n], attack_masks[n]) for n, k in
                                zip(attack_index, range(n_attack))], dtype=metadata_type_attack)
    attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

    out_file.flush()
    out_file.close()

    # compute_snr("OPOI")


def generate_nopoi(window, desync=False):
    n_profiling = 70000
    n_attack = 10000
    number_of_samples = 150000

    trace_folder = "/"
    in_file = h5py.File(f'{trace_folder}dpa_v42_full.h5', 'r')
    profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)[:, 150000:]
    attack_samples = np.array(in_file['Attack_traces/traces'], dtype=np.float32)[:, 150000:]
    profiling_key = in_file['Profiling_traces/metadata']['key']
    attack_key = in_file['Attack_traces/metadata']['key']
    profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
    attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
    profiling_ciphertext = in_file['Profiling_traces/metadata']['ciphertext']
    attack_ciphertext = in_file['Attack_traces/metadata']['ciphertext']
    profiling_masks = in_file['Profiling_traces/metadata']['masks']
    attack_masks = in_file['Attack_traces/metadata']['masks']

    ns = int(len(profiling_samples[0]) / window) * 2

    attack_samples_nopoi = np.zeros((n_attack, ns))
    profiling_samples_nopoi = np.zeros((n_profiling, ns))

    for trace_index in tqdm(range(n_profiling)):
        if desync:
            shift = random.randint(-50, 50)
            trace_tmp_shifted = np.zeros(number_of_samples)
            if shift > 0:
                trace_tmp_shifted[0:number_of_samples - shift] = profiling_samples[trace_index][shift:number_of_samples]
                trace_tmp_shifted[number_of_samples - shift:number_of_samples] = profiling_samples[trace_index][0:shift]
            else:
                trace_tmp_shifted[0:abs(shift)] = profiling_samples[trace_index][number_of_samples - abs(shift):number_of_samples]
                trace_tmp_shifted[abs(shift):number_of_samples] = profiling_samples[trace_index][0:number_of_samples - abs(shift)]
            profiling_samples[trace_index] = trace_tmp_shifted

        profiling_samples_nopoi[trace_index] = winres(profiling_samples[trace_index], window=window)

    for trace_index in tqdm(range(n_attack)):
        if desync:
            shift = random.randint(-50, 50)
            trace_tmp_shifted = np.zeros(number_of_samples)
            if shift > 0:
                trace_tmp_shifted[0:number_of_samples - shift] = attack_samples[trace_index][shift:number_of_samples]
                trace_tmp_shifted[number_of_samples - shift:number_of_samples] = attack_samples[trace_index][0:shift]
            else:
                trace_tmp_shifted[0:abs(shift)] = attack_samples[trace_index][number_of_samples - abs(shift):number_of_samples]
                trace_tmp_shifted[abs(shift):number_of_samples] = attack_samples[trace_index][0:number_of_samples - abs(shift)]
            attack_samples[trace_index] = trace_tmp_shifted

        attack_samples_nopoi[trace_index] = winres(attack_samples[trace_index], window=window)

    profiling_index = [n for n in range(n_profiling)]
    attack_index = [n for n in range(n_attack)]

    if desync:
        out_file = h5py.File(f'{trace_folder}DPAV42_nopoi/dpa_v42_nopoi_window_{window}_desync.h5', 'w')
    else:
        out_file = h5py.File(f'{trace_folder}DPAV42_nopoi/dpa_v42_nopoi_window_{window}.h5', 'w')
    profiling_traces_group = out_file.create_group("Profiling_traces")
    attack_traces_group = out_file.create_group("Attack_traces")

    profiling_traces_group.create_dataset(name="traces", data=profiling_samples_nopoi, dtype=profiling_samples_nopoi.dtype)
    attack_traces_group.create_dataset(name="traces", data=attack_samples_nopoi, dtype=attack_samples_nopoi.dtype)

    metadata_type_profiling = np.dtype([("plaintext", profiling_plaintext.dtype, (len(profiling_plaintext[0]),)),
                                        ("ciphertext", profiling_ciphertext.dtype, (len(profiling_ciphertext[0]),)),
                                        ("key", profiling_key.dtype, (len(profiling_key[0]),)),
                                        ("masks", profiling_masks.dtype, (len(profiling_masks[0]),))])
    metadata_type_attack = np.dtype([("plaintext", attack_plaintext.dtype, (len(attack_plaintext[0]),)),
                                     ("ciphertext", attack_ciphertext.dtype, (len(attack_ciphertext[0]),)),
                                     ("key", attack_key.dtype, (len(attack_key[0]),)),
                                     ("masks", attack_masks.dtype, (len(attack_masks[0]),))])

    profiling_metadata = np.array([(profiling_plaintext[n], profiling_ciphertext[n], profiling_key[n], profiling_masks[n]) for n, k in
                                   zip(profiling_index, range(n_profiling))], dtype=metadata_type_profiling)
    profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

    attack_metadata = np.array([(attack_plaintext[n], attack_ciphertext[n], attack_key[n], attack_masks[n]) for n, k in
                                zip(attack_index, range(n_attack))], dtype=metadata_type_attack)
    attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

    out_file.flush()
    out_file.close()


def generate_nopoi_15000():
    ns = 15000
    n_profiling = 70000
    n_attack = 10000

    in_file = h5py.File(f'{directory_dataset["NOPOI"]}/dpa_v42_nopoi.h5', "r")
    profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
    attack_samples = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
    profiling_key = in_file['Profiling_traces/metadata']['key']
    attack_key = in_file['Attack_traces/metadata']['key']
    profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
    attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
    profiling_ciphertext = in_file['Profiling_traces/metadata']['ciphertext']
    attack_ciphertext = in_file['Attack_traces/metadata']['ciphertext']
    profiling_masks = in_file['Profiling_traces/metadata']['masks']
    attack_masks = in_file['Attack_traces/metadata']['masks']

    profiling_samples = profiling_samples[:, ns:]
    attack_samples = attack_samples[:, ns:]

    profiling_index = [n for n in range(n_profiling)]
    attack_index = [n for n in range(n_attack)]

    trace_folder = "/"
    out_file = h5py.File(f'{trace_folder}DPAV42_nopoi/dpa_v42_nopoi_15000.h5', 'w')

    profiling_traces_group = out_file.create_group("Profiling_traces")
    attack_traces_group = out_file.create_group("Attack_traces")

    profiling_traces_group.create_dataset(name="traces", data=profiling_samples, dtype=profiling_samples.dtype)
    attack_traces_group.create_dataset(name="traces", data=attack_samples, dtype=attack_samples.dtype)

    metadata_type_profiling = np.dtype([("plaintext", profiling_plaintext.dtype, (len(profiling_plaintext[0]),)),
                                        ("ciphertext", profiling_ciphertext.dtype, (len(profiling_ciphertext[0]),)),
                                        ("key", profiling_key.dtype, (len(profiling_key[0]),)),
                                        ("masks", profiling_masks.dtype, (len(profiling_masks[0]),))])
    metadata_type_attack = np.dtype([("plaintext", attack_plaintext.dtype, (len(attack_plaintext[0]),)),
                                     ("ciphertext", attack_ciphertext.dtype, (len(attack_ciphertext[0]),)),
                                     ("key", attack_key.dtype, (len(attack_key[0]),)),
                                     ("masks", attack_masks.dtype, (len(attack_masks[0]),))])

    profiling_metadata = np.array([(profiling_plaintext[n], profiling_ciphertext[n], profiling_key[n], profiling_masks[n]) for n, k in
                                   zip(profiling_index, range(n_profiling))], dtype=metadata_type_profiling)
    profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

    attack_metadata = np.array([(attack_plaintext[n], attack_ciphertext[n], attack_key[n], attack_masks[n]) for n, k in
                                zip(attack_index, range(n_attack))], dtype=metadata_type_attack)
    attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

    out_file.flush()
    out_file.close()

    # compute_snr("NOPOI")


def merge_dataset():
    n_profiling = 70000
    n_attack = 10000

    print("opening dpa_v42_0_100000")
    trace_folder = "/"
    in_file = h5py.File(f'{trace_folder}dpa_v42_0_100000.h5', 'r')
    plaintexts = in_file['Attack_traces/metadata']['plaintext']
    ciphertexts = in_file['Attack_traces/metadata']['ciphertext']
    masks = in_file['Attack_traces/metadata']['masks']
    keys = in_file['Attack_traces/metadata']['key']

    attack_plaintexts = plaintexts[n_profiling: n_attack + n_profiling]
    attack_ciphertexts = ciphertexts[n_profiling: n_attack + n_profiling]
    attack_masks = masks[n_profiling: n_attack + n_profiling]
    attack_keys = keys[n_profiling: n_attack + n_profiling]

    profiling_plaintexts = plaintexts[:n_profiling]
    profiling_ciphertexts = ciphertexts[:n_profiling]
    profiling_masks = masks[:n_profiling]
    profiling_keys = keys[:n_profiling]

    ns_chunck = 100000
    attack_samples = np.zeros((n_attack, 300000))
    profiling_samples = np.zeros((n_profiling, 300000))

    for s_i in range(1, 4):
        fs = ns_chunck * s_i

        print(f"opening dpa_v42_{fs}_{fs + ns_chunck} ...")
        in_file = h5py.File(f"{trace_folder}dpa_v42_{fs}_{fs + ns_chunck}.h5", 'r')
        profiling_samples[:, (s_i - 1) * ns_chunck: s_i * ns_chunck] = np.array(in_file['Attack_traces/traces'], dtype=np.float32)[
                                                                       :n_profiling]
        attack_samples[:, (s_i - 1) * ns_chunck: s_i * ns_chunck] = np.array(in_file['Attack_traces/traces'], dtype=np.float32)[
                                                                    n_profiling:]

        print(f"opening dpa_v42_{fs}_{fs + ns_chunck} ...done")

    profiling_index = [n for n in range(n_profiling)]
    attack_index = [n for n in range(n_attack)]

    out_file = h5py.File(f'{trace_folder}dpa_v42_full.h5', 'w')

    profiling_traces_group = out_file.create_group("Profiling_traces")
    attack_traces_group = out_file.create_group("Attack_traces")

    profiling_traces_group.create_dataset(name="traces", data=profiling_samples, dtype=profiling_samples.dtype)
    attack_traces_group.create_dataset(name="traces", data=attack_samples, dtype=attack_samples.dtype)

    metadata_type_profiling = np.dtype([("plaintext", profiling_plaintexts.dtype, (len(profiling_plaintexts[0]),)),
                                        ("ciphertext", profiling_ciphertexts.dtype, (len(profiling_ciphertexts[0]),)),
                                        ("key", profiling_keys.dtype, (len(profiling_keys[0]),)),
                                        ("masks", profiling_masks.dtype, (len(profiling_masks[0]),))])
    metadata_type_attack = np.dtype([("plaintext", attack_plaintexts.dtype, (len(attack_plaintexts[0]),)),
                                     ("ciphertext", attack_ciphertexts.dtype, (len(attack_ciphertexts[0]),)),
                                     ("key", attack_keys.dtype, (len(attack_keys[0]),)),
                                     ("masks", attack_masks.dtype, (len(attack_masks[0]),))])

    profiling_metadata = np.array([(profiling_plaintexts[n], profiling_ciphertexts[n], profiling_keys[n], profiling_masks[n]) for n, k in
                                   zip(profiling_index, range(n_profiling))], dtype=metadata_type_profiling)
    profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

    attack_metadata = np.array([(attack_plaintexts[n], attack_ciphertexts[n], attack_keys[n], attack_masks[n]) for n, k in
                                zip(attack_index, range(n_attack))], dtype=metadata_type_attack)
    attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

    out_file.flush()
    out_file.close()


if __name__ == "__main__":
    merge_dataset()
    generate_nopoi(10)
    generate_nopoi(20)
    generate_nopoi(40)
    generate_nopoi(80)
    generate_nopoi(10, desync=True)
    generate_nopoi(20, desync=True)
    generate_nopoi(40, desync=True)
    generate_nopoi(80, desync=True)
    generate_opoi()

    generate_rpoi()
    generate_rpoi(gaussian_noise=3)
    generate_rpoi(gaussian_noise=10)
