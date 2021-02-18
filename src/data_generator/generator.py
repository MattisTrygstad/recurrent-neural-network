

import math
from ntpath import split
import h5py
import random
from typing import Callable
import numpy as np
from data_generator.evaluator import Evaluator

from enums import DatasetType, ImageType
from utils.progress import print_progress
from utils.shuffle_arrays import shuffle_arrays


class Generator():

    def __init__(self) -> None:
        # Equilateral Triangle
        # ((x1, y1), (x2, y2), (x3, y3))
        self.relative_triangle_coordinates = ((0.5, 0.75), (0, 0.25), (1, 0.25))

        # ((x_min, y_min), (x_max, y_max))
        self.relative_rectangle_coordinates = ((0, 0), (1, 1))

        # ((center_x, center_y), (point_on_arc_x, point_on_arc_y))
        self.relative_circle_coordinates = ((0.5, 0.5), (0.8, 0.8))

    def generate_dataset(self, dataset_size: int, canvas_size: int = 50, noise_ratio: float = 0.05, pos_deviaiton: int = 0.2, line_width_deviaiton: float = 0.2, width_range: tuple = (10, 40), height_range: tuple = (10, 40), split_ratios: tuple = (0.7, 0.2, 0.1)) -> None:
        training_samples = math.floor(dataset_size * split_ratios[0])
        validation_samples = math.floor(dataset_size * split_ratios[1])
        test_samples = dataset_size - training_samples - validation_samples

        dataset_samples = list((training_samples, validation_samples, test_samples))

        total_samples = sum(dataset_samples)
        progress = 0
        for dataset_type in DatasetType:
            dataset_index = dataset_type.value

            samples = dataset_samples[dataset_index]

            image_data_shape = (samples, canvas_size, canvas_size)
            label_shape = (samples,)

            image_data = np.ndarray(shape=image_data_shape)
            labels = np.ndarray(shape=label_shape)

            for x in range(dataset_samples[dataset_index]):
                image_height = random.uniform(height_range[0], height_range[1])
                image_width = random.uniform(width_range[0], width_range[1])
                deviation = (random.uniform(-pos_deviaiton, pos_deviaiton) * canvas_size, random.uniform(-pos_deviaiton, pos_deviaiton) * canvas_size)
                line_width_deviation = random.uniform(-line_width_deviaiton, line_width_deviaiton)

                type_index = x % len(list(ImageType))
                type = ImageType(type_index)

                data, type = self.generate_image(canvas_size, image_height, image_width, type, noise_ratio, deviation, line_width_deviation)

                image_data[x] = data
                labels[x] = type_index

                progress += 1
                print_progress(progress, total_samples, 'Generating dataset: ', f'{progress}/{total_samples}')

            # Shuffle data with same permutaion for image_data and labels
            image_data, labels = shuffle_arrays(image_data, labels)

            # Save data
            h5f = h5py.File(f'../data/{dataset_type.name.lower()}.h5', 'w')
            h5f.create_dataset(name='image_data', data=image_data, shape=image_data_shape, dtype='float64', chunks=image_data_shape, maxshape=(None, None, None))
            h5f.create_dataset(name='labels', data=labels, shape=label_shape, dtype='int', chunks=label_shape, maxshape=(None,))
            h5f.close()

    def generate_image(self, canvas_size: int, image_height: int, image_width: int, type: ImageType, noise_ratio: float, deviation: tuple, line_width_deviation: float) -> None:
        grid_data = np.zeros((canvas_size, canvas_size))

        if type == ImageType.TRIANGLE:
            self.generate_shape(grid_data, Evaluator.triangle_evaluation, type, canvas_size, image_height, image_width, noise_ratio, deviation, line_width_deviation)

        if type == ImageType.RECTANGLE:
            self.generate_shape(grid_data, Evaluator.rectangle_evaluation, type, canvas_size, image_height, image_width, noise_ratio, deviation, line_width_deviation)

        if type == ImageType.EMPTY_CIRCLE:
            self.generate_shape(grid_data, Evaluator.empty_circle_evaluation, type, canvas_size, image_height, image_width, noise_ratio, deviation, line_width_deviation)

        if type == ImageType.FILLED_CIRCLE:
            self.generate_shape(grid_data, Evaluator.filled_circle_evaluation, type, canvas_size, image_height, image_width, noise_ratio, deviation, line_width_deviation)

        return grid_data, type

    def generate_shape(self, grid_data: np.ndarray, evaluation_method: Callable, type: ImageType, canvas_size: int, image_height: int, image_width: int, noise_ratio: float, deviation: tuple, line_width_deviation: float) -> bool:
        p1, p2, p3, radius, line_width = self.generate_absolute_coordinates(type, canvas_size, image_height, image_width, deviation, line_width_deviation)

        for x in range(canvas_size):
            for y in range(canvas_size):
                if evaluation_method(x, y, canvas_size=canvas_size, p1=p1, p2=p2, p3=p3, radius=radius, line_width=line_width):
                    grid_data[y][x] = 1

                # Random noise
                if random.uniform(0, 1) < noise_ratio:
                    grid_data[y][x] = 0 if grid_data[y][x] == 1 else 1

    def generate_absolute_coordinates(self, type: ImageType, size: int, image_height: int, image_width: int, deviation: tuple, line_width_deviation: float) -> tuple:
        # Points used for generating various shapes
        p1, p2, p3 = None, None, None

        x_deviation, y_deviation = deviation

        # Circle attributes
        radius = None
        line_width = None

        absolute_points = []
        if type == ImageType.TRIANGLE:
            # Generate absolute coordinates
            for point in self.relative_triangle_coordinates:
                absolute_points.append((point[0] * image_width, point[1] * image_height))

            ((x1, y1), (x2, y2), (x3, y3)) = tuple(absolute_points)
            center_x = (x1 + x2 + x3) / 3
            center_y = (y1 + y2 + y3) / 3

            # Center figure in canvas
            centered_points = [((point[0] + size / 2 - center_x + x_deviation), (point[1] + size / 2 - center_y + y_deviation)) for point in absolute_points]
            p1, p2, p3 = tuple(centered_points)

        elif type == ImageType.RECTANGLE:
            # Generate absolute coordinates
            for point in self.relative_rectangle_coordinates:
                absolute_points.append((point[0] * image_width, point[1] * image_height))

            ((x1, y1), (x2, y2)) = tuple(absolute_points)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Center figure in canvas
            centered_points = [((point[0] + size / 2 - center_x + x_deviation), (point[1] + size / 2 - center_y + y_deviation)) for point in absolute_points]

            p1, p2 = tuple(centered_points)

        elif type == ImageType.EMPTY_CIRCLE or type == ImageType.FILLED_CIRCLE:
            for point in self.relative_circle_coordinates:
                absolute_points.append((point[0] * image_width, point[1] * image_height))

            (center_x, center_y), (arc_x, arc_y) = tuple(absolute_points)

            radius = math.sqrt((arc_x - center_x)**2 + (arc_y - center_y)**2)

            # Used for empty circle
            line_width = radius * 2 * (1 + line_width_deviation)

            # Circle center
            p1 = (size / 2 + x_deviation, size / 2 + y_deviation)

        # print(p1, p2, p3, radius)
        return p1, p2, p3, radius, line_width
