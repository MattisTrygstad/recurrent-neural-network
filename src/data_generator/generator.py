
import math
import random
import sys
import h5py
import numpy as np
from enums import DatasetType
from utils.config_parser import Config

from utils.progress import print_progress
from utils.shuffle_arrays import shuffle_arrays


class Generator():

    @staticmethod
    def generate_dataset(shift_rules: list, vector_length: int, sequence_length: int, dataset_size: int, probability: float, split_ratios: tuple = (0.7, 0.2, 0.1)) -> None:
        training_samples = math.floor(dataset_size * split_ratios[0])
        validation_samples = math.floor(dataset_size * split_ratios[1])
        test_samples = dataset_size - training_samples - validation_samples

        dataset_samples = list((training_samples, validation_samples, test_samples))
        total_samples = sum(dataset_samples)
        progress = 0

        for dataset_type in DatasetType:
            dataset_index = dataset_type.value

            samples = dataset_samples[dataset_index]

            inputs = np.ndarray((samples, sequence_length, vector_length))
            outputs = np.ndarray((samples, sequence_length, vector_length))
            rules = np.ndarray((samples))

            for index in range(dataset_samples[dataset_index]):
                input, output, rule = Generator.__generate_sequence(shift_rules, vector_length, sequence_length, probability)
                inputs[index] = input
                outputs[index] = output
                rules[index] = rule

                progress += 1
                print_progress(progress, total_samples, 'Generating dataset: ', f'{progress}/{total_samples}')

            h5f = h5py.File(f'../data/{dataset_type.name.lower()}.h5', 'w')
            h5f.create_dataset(name='inputs', data=inputs, shape=inputs.shape, dtype='int', chunks=inputs.shape, maxshape=(None, None, None))
            h5f.create_dataset(name='outputs', data=outputs, shape=outputs.shape, dtype='int', chunks=outputs.shape, maxshape=(None, None, None))
            h5f.create_dataset(name='rules', data=rules, shape=rules.shape, dtype='int', chunks=rules.shape, maxshape=(None,))
            h5f.close()

    @staticmethod
    def __generate_vector(vector_length: int, probability: float) -> np.ndarray:
        vector = np.zeros(vector_length)
        ones = int(round(vector_length * probability))
        vector[:ones] = 1
        if Config.shuffle_pattern:
            np.random.shuffle(vector)
        return vector

    @staticmethod
    def __generate_sequence(shift_rules: list, vector_length: int, sequence_length: int, probability: float) -> np.ndarray:
        initial_pattern = Generator.__generate_vector(vector_length, probability)

        rule = random.choice(shift_rules)
        input = np.ndarray((sequence_length, vector_length), 'int')
        output = np.ndarray((sequence_length, vector_length), 'int')
        input[0] = initial_pattern

        for x in range(1, sequence_length):
            input[x] = np.roll(input[x - 1], rule)
            output[x - 1] = input[x]

        output[sequence_length - 1] = np.roll(input[sequence_length - 1], rule)

        return input, output, rule
