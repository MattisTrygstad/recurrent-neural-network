from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    """
    Abstract class for Activation Functions
    """

    def __init__(self) -> None:
        self.activations = None

    @abstractmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dA: np.ndarray) -> np.ndarray:
        pass
