
import numpy as np
from abstract_classes.activation_function import ActivationFunction


class Linear(ActivationFunction):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return Z

    def backward(self, dA: np.ndarray) -> np.ndarray:
        return dA * np.ones_like(dA)
