
from abc import ABC, abstractmethod
import numpy as np


class Regularizer(ABC):
    """
    Abstract class for Regularizers
    """

    def __init__(self, regularization_rate: float) -> None:
        self.regularization_rate = regularization_rate

    @abstractmethod
    def compute_loss(self) -> float:
        pass

    @abstractmethod
    def regularize(self, input: np.ndarray) -> np.ndarray:
        pass
