from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    """
    Abstract class for Activation Functions
    """

    @abstractmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, ds: np.ndarray) -> np.ndarray:
        pass
