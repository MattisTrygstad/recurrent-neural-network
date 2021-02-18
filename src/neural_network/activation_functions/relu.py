
import sys
import numpy as np
from abstract_classes.activation_function import ActivationFunction
from abstract_classes.layer import Layer


class Relu(ActivationFunction):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.activations = Z * (Z > 0)
        return self.activations

    def backward(self, dA: np.ndarray) -> np.ndarray:
        gradient = (self.activations > 0) * dA
        return gradient
