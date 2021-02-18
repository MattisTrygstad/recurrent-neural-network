
import numpy as np
from abstract_classes.loss_function import LossFunction


class CrossEntropy(LossFunction):
    @staticmethod
    def compute_loss(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        return -1 * np.sum(target * np.log(prediction))

    @staticmethod
    def compute_loss_derivative(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.where(prediction != 0, -target / prediction, 0.0)
