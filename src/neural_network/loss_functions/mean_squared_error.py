import numpy as np
from abstract_classes.loss_function import LossFunction


class MeanSquaredError(LossFunction):

    @staticmethod
    def compute_loss(prediction: np.ndarray, target: np.ndarray):
        prediction = np.transpose(prediction)
        assert prediction.shape == target.shape

        losses = []
        batch_size = prediction.shape[0]
        vector_size = prediction.shape[1]

        for batch_index in range(batch_size):
            loss = (np.square(np.subtract(prediction[batch_index], target[batch_index]))).mean(axis=None)
            losses.append(loss / vector_size)

        return np.sum(losses) / batch_size

    @staticmethod
    def compute_loss_derivative(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert prediction.shape == target.shape
        loss_derivatives = []
        batch_size = prediction.shape[0]

        for batch_index in range(batch_size):
            dA = 2 * (prediction[batch_index] - target[batch_index])
            loss_derivatives.append(dA)

        # TODO: remove index for batch support
        return loss_derivatives[0]
