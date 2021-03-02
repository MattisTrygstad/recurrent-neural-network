
import sys
import numpy as np
from abstract_classes.loss_function import LossFunction


class CrossEntropy(LossFunction):
    @staticmethod
    def compute_loss(prediction: np.ndarray, target: np.ndarray) -> float:
        prediction = np.transpose(prediction)
        assert prediction.shape == target.shape
        losses = []

        batch_size = prediction.shape[0]
        vector_size = prediction.shape[1]
        # print(prediction.shape)
        # print(target.shape)
        for batch_index in range(batch_size):
            # print(target[batch_index])
            loss = -np.sum(target[batch_index][i] * np.log2(prediction[batch_index][i]) for i in range(vector_size))
            losses.append(loss / vector_size)

        return sum(losses) / batch_size

    @staticmethod
    def compute_loss_derivative(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert prediction.shape == target.shape
        loss_derivatives = []
        batch_size = prediction.shape[0]

        for batch_index in range(batch_size):
            dA = np.where(prediction[batch_index] != 0, -target[batch_index] / prediction[batch_index], 0.0)
            loss_derivatives.append(dA)

        # TODO: remove index for batch support
        return loss_derivatives[0]
