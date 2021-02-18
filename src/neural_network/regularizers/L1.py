
from abstract_classes.regularizer import Regularizer
import numpy as np


class L1(Regularizer):

    def __init__(self, regularization_rate: float) -> None:
        super().__init__(regularization_rate)

    def compute_loss(self, input: np.ndarray) -> float:
        return self.regularization_rate * input.sum()

    def regularize(self, input: np.ndarray) -> np.ndarray:
        return self.regularization_rate * np.ones(input.shape)
