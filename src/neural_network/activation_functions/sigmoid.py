
import numpy as np
from abstract_classes.activation_function import ActivationFunction


class Sigmoid(ActivationFunction):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.activations = 1 / (1 + np.exp(-Z))
        return self.activations

    def backward(self, dA: np.ndarray) -> np.ndarray:
        return dA * (self.activations * (1 - self.activations))
