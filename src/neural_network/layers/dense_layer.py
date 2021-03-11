import numpy as np
from abstract_classes.activation_function import ActivationFunction
from abstract_classes.layer import Layer
from abstract_classes.loss_function import LossFunction
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

        # Forward prop params
        self.activated_sums = []
        self.activated_sums_prev_layer = []

        if name:
            self.name = name
        else:
            self.name = f'dense{self.output_shape}'

    def forward_pass(self, input: np.ndarray, add_biases: bool) -> np.ndarray:
        # Send each case through the network from input to output
        activated_sum_prev_layer = self.previous_layer.forward_pass(input, add_biases)
        self.activated_sums_prev_layer.append(activated_sum_prev_layer)

        # Multiply the outputs of the previous layer with the weights
        weighted_sum: np.ndarray = np.transpose(self.weights) @ np.transpose(activated_sum_prev_layer)

        # Add biases
        if add_biases:
            repeats = weighted_sum.shape[-1]
            new_biases = np.repeat(self.biases, repeats, axis=-1)
            weighted_sum += new_biases

        # Apply activation function
        activated_sum = np.transpose(self.activation_func.forward(weighted_sum))

        self.activated_sums.append(activated_sum)

        return activated_sum

    def backward_pass(self, output_jacobian: np.ndarray) -> float:
        # print('backprop output layer')

        # O_k
        activated_sum = self.activated_sums.pop()
        # H_k
        activated_sum_prev_layer = self.activated_sums_prev_layer.pop()

        batch_size = output_jacobian.shape[0]

        V_grad_prev_seq = np.zeros_like(self.weights) if not hasattr(self, 'V_grad_cumulative') else self.V_grad_cumulative

        # TODO: Comment shapes
        self.V_grad_cumulative = self.compute_weight_gradient(output_jacobian, self.activation_func, activated_sum, activated_sum_prev_layer, batch_size, V_grad_prev_seq)
        # print('V_grad', V_grad.shape)

        neighbor_jacobian = self.compute_neighbor_jacobian(self.activation_func, activated_sum, self.weights, batch_size)

        # print('output_jacobian', output_jacobian.shape)
        # print('neighbor_jacobian', neighbor_jacobian.shape)
        next_output_jacobian = output_jacobian @ neighbor_jacobian

        # print('return next_output_jacobian', next_output_jacobian.shape)
        return self.previous_layer.backward_pass(next_output_jacobian)
