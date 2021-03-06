
import numpy as np
from abstract_classes.activation_function import ActivationFunction


class Tanh(ActivationFunction):

    def forward(self, unactivated_sum: np.ndarray) -> np.ndarray:
        return np.tanh(unactivated_sum)

    def backward(self, ds: np.ndarray) -> np.ndarray:
        return (1 - self.activations**2) * ds
