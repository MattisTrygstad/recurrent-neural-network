
import sys
import numpy as np
from abstract_classes.activation_function import ActivationFunction
from abstract_classes.layer import Layer


class Softmax(ActivationFunction):

    def __init__(self) -> None:
        self.activations = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.activations = np.exp(Z - np.max(Z, axis=-1, keepdims=True)) / np.exp(Z - np.max(Z, axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)
        return self.activations

    def backward(self, dA: np.ndarray) -> np.ndarray:
        act_shape = self.activations.shape
        act = self.activations.reshape(act_shape[0], 1, act_shape[-1])

        jacobian = - (act.transpose((0, 2, 1)) @ act) * (1 - np.identity(self.activations.shape[-1]))
        jacobian += np.identity(act_shape[-1]) * (act * (1 - act)).transpose((0, 2, 1))

        dZ = (jacobian @ dA.reshape(act_shape[0], act_shape[-1], 1))
        dZ = dZ.reshape((act_shape[0], act_shape[-1]))

        return dZ
