import sys
import numpy as np
from abstract_classes.activation_function import ActivationFunction
from abstract_classes.layer import Layer
from abstract_classes.regularizer import Regularizer


class DenseLayer(Layer):

    def __init__(self, output_shape: int, previous_layer: Layer, activation_func: ActivationFunction, learning_rate: float, regularizer: Regularizer, init_weight_range: tuple = (-0.1, 0.1), name: str = None) -> None:
        super().__init__(output_shape)

        self.previous_layer = previous_layer
        self.input_shape = previous_layer.output_shape
        self.biases = np.zeros((output_shape, 1))
        self.weights = np.random.uniform(low=init_weight_range[0], high=init_weight_range[1], size=(self.input_shape, self.output_shape))
        self.activation_func = activation_func
        self.learning_rate = learning_rate
        self.regularizer = regularizer

        # Stored forward prop output for current layer, used in back prop
        self.Z = None

        if name:
            self.name = name
        else:
            self.name = f'dense{self.input_shape}'

    def forward_pass(self, input: np.ndarray, add_biases: bool) -> np.ndarray:
        # Send each case through the network from input to output
        self.A_previous_layer = self.previous_layer.forward_pass(input, add_biases)

        # Multiply the outputs of the previous layer with the weights
        weighted_sum: np.ndarray = np.transpose(self.weights) @ self.A_previous_layer.transpose()

        # Add biases
        if add_biases:
            repeats = weighted_sum.shape[-1]
            new_biases = np.repeat(self.biases, repeats, axis=-1)
            weighted_sum += new_biases

        self.Z = weighted_sum.transpose()
        # Apply activation function
        A = self.activation_func.forward(self.Z)
        return A

    def backward_pass(self, dA: np.ndarray) -> float:
        # dA ~ derivative of losses
        dZ = self.activation_func.backward(dA)

        dW = (self.A_previous_layer.transpose() @ dZ) / dZ.shape[0]
        db = (dZ.transpose().sum(axis=-1, keepdims=True)) / dZ.shape[0]

        regularizer_loss = 0
        if self.regularizer is not None:
            regularizer_loss = self.regularizer.compute_loss(self.weights) + self.regularizer.compute_loss(self.biases)
            self.weights -= self.learning_rate * self.regularizer.regularize(self.weights)
            self.biases -= self.learning_rate * self.regularizer.regularize(self.biases)

        self.weights -= self.learning_rate * dW
        self.biases -= self.learning_rate * db

        dA_next = np.transpose(self.weights @ np.transpose(dZ))
        return regularizer_loss + self.previous_layer.backward_pass(dA_next)
