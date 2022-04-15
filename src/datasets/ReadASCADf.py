import numpy as np
import h5py
from tensorflow.keras.utils import *

aes_sbox = np.array([
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


class ReadASCADf:

    def __init__(self, n_profiling, n_validation, n_attack, file_path="", target_byte=2, leakage_model="ID", first_sample=0,
                 number_of_samples=700, reshape_to_cnn=False):
        self.n_profiling = n_profiling
        self.n_validation = n_validation
        self.n_attack = n_attack
        self.target_byte = target_byte
        self.leakage_model = leakage_model
        self.file_path = file_path
        self.fs = first_sample
        self.ns = number_of_samples
        self.classes = 9 if leakage_model == "HW" else 256

        self.x_profiling = []
        self.x_validation = []
        self.x_attack = []

        self.y_profiling = []
        self.y_validation = []
        self.y_attack = []

        self.profiling_labels = []
        self.validation_labels = []
        self.attack_labels = []

        self.labels_key_hypothesis_validation = None
        self.labels_key_hypothesis_attack = None

        self.round_key = "4DFBE0F27221FE10A78D4ADC8E490469"

        self.correct_key = bytearray.fromhex(self.round_key)[target_byte]

        self.load_dataset(reshape_to_cnn=reshape_to_cnn)

    def load_dataset(self, reshape_to_cnn=False):
        in_file = h5py.File(self.file_path, "r")

        profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
        attack_samples = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
        profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
        attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
        profiling_key = in_file['Profiling_traces/metadata']['key']
        attack_key = in_file['Attack_traces/metadata']['key']

        profiling_plaintexts = profiling_plaintext[:]
        profiling_keys = profiling_key[:]
        validation_plaintexts = attack_plaintext[:self.n_validation]
        validation_keys = attack_key[:self.n_validation]
        attack_plaintexts = attack_plaintext[self.n_validation:self.n_validation + self.n_attack]
        attack_keys = attack_key[self.n_validation:self.n_validation + self.n_attack]

        x_profiling = profiling_samples[0:self.n_profiling, self.fs:self.fs + self.ns]
        x_validation = attack_samples[0:self.n_validation, self.fs:self.fs + self.ns]
        x_attack = attack_samples[self.n_validation:self.n_validation + self.n_attack, self.fs:self.fs + self.ns]

        z_score_mean, z_score_std = self.create_z_score_norm(x_profiling)
        self.apply_z_score_norm(x_profiling, z_score_mean, z_score_std)
        self.apply_z_score_norm(x_validation, z_score_mean, z_score_std)
        self.apply_z_score_norm(x_attack, z_score_mean, z_score_std)

        self.x_profiling = x_profiling.astype('float32')
        self.x_validation = x_validation.astype('float32')
        self.x_attack = x_attack.astype('float32')

        if reshape_to_cnn:
            self.x_profiling = self.x_profiling.reshape((self.x_profiling.shape[0], self.x_profiling.shape[1], 1))
            self.x_validation = self.x_validation.reshape((self.x_validation.shape[0], self.x_validation.shape[1], 1))
            self.x_attack = self.x_attack.reshape((self.x_attack.shape[0], self.x_attack.shape[1], 1))

        self.profiling_labels = self.aes_labelize(profiling_plaintexts, profiling_keys)
        self.validation_labels = self.aes_labelize(validation_plaintexts, validation_keys)
        self.attack_labels = self.aes_labelize(attack_plaintexts, attack_keys)

        self.y_profiling = to_categorical(self.profiling_labels, num_classes=self.classes)
        self.y_validation = to_categorical(self.validation_labels, num_classes=self.classes)
        self.y_attack = to_categorical(self.attack_labels, num_classes=self.classes)

        self.labels_key_hypothesis_validation = self.create_labels_key_guess(validation_plaintexts)
        self.labels_key_hypothesis_attack = self.create_labels_key_guess(attack_plaintexts)

    def create_z_score_norm(self, dataset):
        z_score_mean = np.mean(dataset, axis=0)
        z_score_std = np.std(dataset, axis=0)
        return z_score_mean, z_score_std

    def apply_z_score_norm(self, dataset, z_score_mean, z_score_std):
        for index in range(len(dataset)):
            dataset[index] = (dataset[index] - z_score_mean) / z_score_std

    def aes_labelize(self, plaintexts, keys):

        if np.array(keys).ndim == 1:
            """ repeat key if argument keys is a single key candidate (for GE and SR computations)"""
            keys = np.full([len(plaintexts), 16], keys)

        plaintext = [row[self.target_byte] for row in plaintexts]
        key = [row[self.target_byte] for row in keys]
        state = [int(p) ^ int(k) for p, k in zip(plaintext, key)]
        intermediates = aes_sbox[state]

        return [bin(iv).count("1") for iv in intermediates] if self.leakage_model == "HW" else intermediates

    def create_labels_key_guess(self, plaintexts):
        labels_key_hypothesis = np.zeros((256, len(plaintexts)), dtype='int64')
        for key_byte_hypothesis in range(256):
            key_h = bytearray.fromhex(self.round_key)
            key_h[self.target_byte] = key_byte_hypothesis
            labels_key_hypothesis[key_byte_hypothesis] = self.aes_labelize(plaintexts, key_h)
        return labels_key_hypothesis
