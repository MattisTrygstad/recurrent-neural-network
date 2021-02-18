
import numpy as np
from abstract_classes.activation_function import ActivationFunction


class Tanh(ActivationFunction):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.activations = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return self.activations

    def backward(self, dA: np.ndarray) -> np.ndarray:
        return dA * (1 - self.activations**2)
