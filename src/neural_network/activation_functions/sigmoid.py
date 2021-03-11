
import numpy as np
from abstract_classes.activation_function import ActivationFunction


class Sigmoid(ActivationFunction):

    def forward(self, unactivated_sum: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-unactivated_sum))

    def backward(self, activated_sum: np.ndarray) -> np.ndarray:
        return (activated_sum * (1 - activated_sum))
