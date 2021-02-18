from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    """
    Abstract class for Loss Functions
    """

    def __init__(self, output_shape: int) -> None:
        self.output_shape = output_shape

    @abstractmethod
    def compute_loss(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def compute_loss_derivative(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass
