import numpy as np
from abstract_classes.loss_function import LossFunction


class MeanSquaredError(LossFunction):

    @staticmethod
    def compute_loss(prediction: np.ndarray, target: np.ndarray):
        return (np.square(np.subtract(prediction, target))).mean(axis=None)

    @staticmethod
    def compute_loss_derivative(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 2 * (prediction - target)
