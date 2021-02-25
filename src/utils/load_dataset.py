import numpy as np
import h5py

from enums import DatasetType


def load_dataset(dataset_name: DatasetType) -> tuple:
    h5f = h5py.File(f'../data/{dataset_name.name.lower()}.h5', 'r')
    inputs = h5f['inputs'][:]
    outputs = h5f['outputs'][:]
    rules = h5f['rules'][:]

    return inputs, outputs, rules
