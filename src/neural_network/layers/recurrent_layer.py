import sys
import numpy as np
from abstract_classes.activation_function import ActivationFunction
from abstract_classes.layer import Layer
from abstract_classes.loss_function import LossFunction
from abstract_classes.regularizer import Regularizer
from utils.config_parser import Config


class RecurrentLayer(Layer):

    def __init__(self, output_shape: int, previous_layer: Layer, activation_func: ActivationFunction, learning_rate: float, init_weight_range: tuple = (-0.1, 0.1), name: str = None) -> None:
        super().__init__(output_shape)

        self.previous_layer = previous_layer
        self.input_shape = previous_layer.output_shape
        self.biases = np.zeros((output_shape, 1))
        self.activation_func = activation_func
        self.learning_rate = learning_rate

        # Parameter calculated during forward prop, used in back prop.
        self.activated_sum_prev_layer = []
        self.activated_sum = []
        self.U_frd = []
        self.W_frd = []

        # Backprop parameters
        self.delta_jacobians = []
        self.W_grads = []
        self.U_grads = []

        # Gradient ~ dU
        self.input_weights = np.random.uniform(low=init_weight_range[0], high=init_weight_range[1], size=(self.input_shape, self.output_shape))

        # Gradient ~ dW
        self.internal_weights = np.random.uniform(low=init_weight_range[0], high=init_weight_range[1], size=(self.output_shape, self.output_shape))

        # Gradient ~ dV
        self.output_weights = np.random.uniform(low=init_weight_range[0], high=init_weight_range[1], size=(self.input_shape, self.output_shape))

        if name:
            self.name = name
        else:
            self.name = f'recurrent{self.input_shape}'

    def forward_pass(self, input: np.ndarray, add_biases: bool) -> np.ndarray:
        print('recurrent forward')
        # Send each case through the network from input to output
        activated_sum_prev_layer = self.previous_layer.forward_pass(input, add_biases)

        activated_sum_prev_seq = np.zeros((1, self.output_shape)) if len(self.activated_sum) == 0 else self.activated_sum[-1]
        # Multiply the outputs of the previous layer with the weights
        print(np.transpose(self.internal_weights).shape)
        print(activated_sum_prev_seq.shape)
        W_frd: np.ndarray = np.transpose(self.internal_weights) @ np.transpose(activated_sum_prev_seq)

        U_frd: np.ndarray = np.transpose(self.input_weights) @ np.transpose(activated_sum_prev_layer)

        temp_sum = W_frd + U_frd

        # Add biases
        if add_biases:
            repeats = W_frd.shape[-1]
            new_biases = np.repeat(self.biases, repeats, axis=-1)
            temp_sum += new_biases

        # Apply activation function
        activated_sum = np.transpose(self.activation_func.forward(temp_sum))

        self.activated_sum_prev_layer.append(activated_sum_prev_layer)
        self.activated_sum.append(activated_sum)
        self.U_frd.append(U_frd)
        self.W_frd.append(W_frd)

        return activated_sum

    def backward_pass(self, output_jacobian: np.ndarray) -> float:
        print('\nbackprop recurrent')
        self.W_frd.pop()
        curr_activated_sum = self.activated_sum[-1]
        prev_activated_sum = np.zeros_like(output_jacobian) if len(self.W_frd) == 0 else self.activated_sum[-2]

        activated_sum_prev_layer = self.activated_sum_prev_layer.pop()

        batch_size = output_jacobian.shape[0]

        # First sequence in backprop
        if len(self.delta_jacobians) == 0:
            # Weigh grad params
            W_grad_prev_seq = np.zeros_like(self.internal_weights)
            U_grad_prev_seq = np.zeros_like(self.input_weights)

            # Delta jacobian params
            recurrent_jacobian = 0
            delta_jacobian = output_jacobian
        else:
            # Weight grad params
            W_grad_prev_seq = self.W_grads[-1]
            U_grad_prev_seq = self.U_grads[-1]

            # Delta jacobian params
            next_activated_sum = self.activated_sum.pop()
            delta_jacobian_prev_seq = self.delta_jacobians[-1]

            recurrent_jacobian = [np.diag((1 - next_activated_sum[x]**2)) @ np.transpose(self.internal_weights) for x in range(batch_size)]
            recurrent_jacobian = np.sum(recurrent_jacobian, axis=0)
            delta_jacobian = output_jacobian + delta_jacobian_prev_seq @ recurrent_jacobian

        self.delta_jacobians.append(delta_jacobian)

        print('delta_jacobian', delta_jacobian.shape)
        print(curr_activated_sum.shape, prev_activated_sum.shape)

        # Shapes: W_grad = W_grad_prev_seq = (recurrent_size, recurrent_size), output_jacobian = (batch_size, bit_vector_size), curr_activated_sum = prev_activated_sum = (batch_size, recurrent_size)
        W_grad = [np.transpose(W_grad_prev_seq) + np.diag(output_jacobian[x]) @ np.outer((1 - curr_activated_sum[x]**2), prev_activated_sum[x]) for x in range(batch_size)]
        W_grad = np.transpose(np.sum(W_grad, axis=0))
        self.W_grads.append(W_grad)

        print('W_grad', W_grad.shape)
        print()

        # TODO: Add shapes
        U_grad = [np.transpose(U_grad_prev_seq) + np.diag(output_jacobian[x]) @ np.outer((1 - curr_activated_sum[x]**2), activated_sum_prev_layer[x]) for x in range(batch_size)]
        U_grad = np.transpose(np.sum(U_grad, axis=0))
        self.U_grads.append(U_grad)
        print('U_grad', U_grad.shape)

        neighbor_jacobian = [np.diag(1 - curr_activated_sum[x]**2) @ np.transpose(self.input_weights) for x in range(batch_size)]
        neighbor_jacobian = np.sum(neighbor_jacobian, axis=0)

        print('neighbor_jacobian', neighbor_jacobian.shape)

        next_output_jacobian = delta_jacobian @ neighbor_jacobian

        print(next_output_jacobian.shape)

        return next_output_jacobian
