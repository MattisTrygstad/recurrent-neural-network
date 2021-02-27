

from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """
    Abstract class for Neural Network layers
    """

    def __init__(self, output_shape: int) -> None:
        self.output_shape = output_shape

    @abstractmethod
    def forward_pass(self, input: np.ndarray, use_biases: bool) -> np.ndarray:
        """
        1. Fetch a minibatch of training cases.
        2. Send each case through the network, from the input to the output layer. At each layer (L), multiply the outputs of the upstream layer by the weights and then add in the biases. Finally, apply the activation function to these sums to produce the outputs of L.
        3. Apply the softmax function to the values entering the output layer to produce the network’s outputs. Remember that softmax has no incoming weights.
        4. Compare the targets to the output values via the loss function.
        5. Cache any information (such as the outputs of each layer) needed for the backward stage.
        """
        pass

    @abstractmethod
    def backward_pass(self, dLo: np.ndarray, input: np.ndarray, diff_s: np.ndarray) -> float:
        """
        1. Compute the initial Jacobian (JSL) representing the derivative of the loss with respect to the network’s (typically softmaxed) outputs.
        2. Pass JSL back through the Softmax layer, modifying it to JNL, which represents the derivative of the loss with respect to the outputs of the layer prior to the softmax, layer N.
        3. Pass JNL to layer N, which uses it to compute its delta Jacobian, δN .
        4. Use δN to compute: a) weight gradients JWL for the incoming weights to N, b) bias gradients JBL for the biases at layer N, and c) JNL−1 to be passed back to layer N-1. 3
        5. Repeat steps 3 and 4 for each layer from N-1 to 1. Nothing needs to be passed back to the Layer 0, the input layer.
        6. After all cases of the minibatch have been passed backwards, and all weight and bias gradients have been computed and accumulated, modify the weights and biases.
        """
        pass
