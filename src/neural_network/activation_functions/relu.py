
import sys
import numpy as np
from abstract_classes.activation_function import ActivationFunction
from abstract_classes.layer import Layer


class Relu(ActivationFunction):

    def forward(self, unactivated_sum: np.ndarray) -> np.ndarray:
        return unactivated_sum * (unactivated_sum > 0)

    def backward(self, activated_sum: np.ndarray) -> np.ndarray:
        return(activated_sum > 0)
