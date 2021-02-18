from enum import Enum


class DatasetType(Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2


class ImageType(Enum):
    TRIANGLE = 0
    EMPTY_CIRCLE = 1
    FILLED_CIRCLE = 2
    RECTANGLE = 3
