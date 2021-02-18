import numpy as np
import h5py

from enums import DatasetType


def load_dataset(dataset_type: DatasetType, flatten: bool = False) -> tuple:
    h5f = h5py.File(f'../data/{dataset_type.name.lower()}.h5', 'r')
    image_data = h5f['image_data'][:]
    labels = h5f['labels'][:]

    if flatten:
        (samples, rows, cols) = image_data.shape
        flattened_image_data = np.ndarray((samples, rows * cols))
        for x in range(len(image_data)):
            flattened_image_data[x] = image_data[x].flatten()

        return flattened_image_data, labels

    else:
        return image_data, labels
