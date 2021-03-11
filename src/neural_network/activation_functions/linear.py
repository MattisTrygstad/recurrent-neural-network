
import numpy as np
from abstract_classes.activation_function import ActivationFunction


class Linear(ActivationFunction):

    def forward(self, unactivated_sum: np.ndarray) -> np.ndarray:
        return unactivated_sum

    def backward(self, activated_sum: np.ndarray) -> np.ndarray:
        return np.ones_like(activated_sum)
