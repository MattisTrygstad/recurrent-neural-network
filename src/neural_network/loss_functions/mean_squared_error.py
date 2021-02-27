import sys
import numpy as np
from abstract_classes.loss_function import LossFunction


class MeanSquaredError(LossFunction):

    @staticmethod
    def compute_loss(prediction: np.ndarray, target: np.ndarray):
        # print(prediction.shape)
        # print(target.shape)
        assert prediction.shape == target.shape

        losses = []
        batch_size = prediction.shape[0]
        vector_size = prediction.shape[1]

        for batch_index in range(batch_size):
            # print(prediction[batch_index])
            # print(target[batch_index])
            loss = np.mean((np.square(prediction[batch_index] - target[batch_index])))
            losses.append(loss / vector_size)

        return losses

    @ staticmethod
    def compute_loss_derivative(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert prediction.shape == target.shape
        loss_derivatives = []
        batch_size = prediction.shape[0]

        for batch_index in range(batch_size):
            dA = 2 * (prediction[batch_index] - target[batch_index])
            loss_derivatives.append(dA)

        return loss_derivatives
