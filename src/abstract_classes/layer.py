

from abc import ABC, abstractmethod
import numpy as np
from abstract_classes.activation_function import ActivationFunction


class Layer(ABC):
    """
    Superclass for Neural Network layers
    """

    def __init__(self, output_shape: int) -> None:
        self.output_shape = output_shape

    def compute_weight_gradient(self, jacobian: np.ndarray, activation_func: ActivationFunction, a1: np.ndarray, a2: np.ndarray, batch_size: int, grad_prev_seq: np.ndarray) -> np.ndarray:
        grad = [np.diag(jacobian[x]) @ np.outer(activation_func.backward(a1[x]), a2[x]) for x in range(batch_size)]
        return np.transpose(np.transpose(grad_prev_seq) + np.sum(grad, axis=0))

    def compute_neighbor_jacobian(self, activation_func: ActivationFunction, a: np.ndarray, weights: np.ndarray, batch_size: int) -> np.ndarray:
        neighbor_jacobian = [np.diag(activation_func.backward(a[x])) @ np.transpose(weights) for x in range(batch_size)]
        return np.sum(neighbor_jacobian, axis=0)

    @abstractmethod
    def forward_pass(self, input: np.ndarray, add_biases: bool) -> np.ndarray:
        """
        1. Fetch a minibatch of training cases.
        2. Send each case through the network, from the input to the output layer. At each layer (L), multiply the outputs of the upstream layer by the weights and then add in the biases. Finally, apply the activation function to these sums to produce the outputs of L.
        3. Apply the softmax function to the values entering the output layer to produce the network’s outputs. Remember that softmax has no incoming weights.
        4. Compare the targets to the output values via the loss function.
        5. Cache any information (such as the outputs of each layer) needed for the backward stage.
        """
        pass

    @abstractmethod
    def backward_pass(self, output_jacobian: np.ndarray) -> float:
        """
        1. Compute the initial Jacobian (JSL) representing the derivative of the loss with respect to the network’s (typically softmaxed) outputs.
        2. Pass JSL back through the Softmax layer, modifying it to JNL, which represents the derivative of the loss with respect to the outputs of the layer prior to the softmax, layer N.
        3. Pass JNL to layer N, which uses it to compute its delta Jacobian, δN .
        4. Use δN to compute: a) weight gradients JWL for the incoming weights to N, b) bias gradients JBL for the biases at layer N, and c) JNL−1 to be passed back to layer N-1. 3
        5. Repeat steps 3 and 4 for each layer from N-1 to 1. Nothing needs to be passed back to the Layer 0, the input layer.
        6. After all cases of the minibatch have been passed backwards, and all weight and bias gradients have been computed and accumulated, modify the weights and biases.
        """
        pass
