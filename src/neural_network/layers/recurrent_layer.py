import sys
import numpy as np
from abstract_classes.activation_function import ActivationFunction
from abstract_classes.layer import Layer
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
        self.init_activated_sum = np.zeros((output_shape, 1))
        self.activated_sum_prev_layer = []
        self.activated_sum = []
        self.U_frd = []
        self.W_frd = []

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
        # TODO: modify to support batch_size != 1 by doing forward pass for each sequence in input.
        # Send each case through the network from input to output
        activated_sum_prev_layer = self.previous_layer.forward_pass(input, add_biases)

        activated_sum_prev_seq = self.init_activated_sum if len(self.activated_sum) == 0 else self.activated_sum[-1]
        # Multiply the outputs of the previous layer with the weights
        W_frd: np.ndarray = self.internal_weights @ activated_sum_prev_seq

        U_frd: np.ndarray = np.transpose(self.input_weights) @ activated_sum_prev_layer

        temp_sum = W_frd + U_frd

        # Add biases
        if add_biases:
            repeats = W_frd.shape[-1]
            new_biases = np.repeat(self.biases, repeats, axis=-1)
            temp_sum += new_biases

        sum = np.transpose(temp_sum)

        # Apply activation function
        activated_sum = np.transpose(self.activation_func.forward(sum))

        self.activated_sum_prev_layer.append(activated_sum_prev_layer)
        self.activated_sum.append(activated_sum)
        self.U_frd.append(U_frd)
        self.W_frd.append(W_frd)
        return activated_sum

    def multiplication_backward(self, weights: np.ndarray, frd: np.ndarray, grad: np.ndarray):
        gradient_weight = grad @ np.transpose(frd)
        chain_gradient = np.transpose(weights) @ grad

        return gradient_weight, chain_gradient

    def add_backward(self, U_frd: np.ndarray, W_frd: np.ndarray, dZ: np.ndarray):
        dx1 = dZ * np.ones_like(U_frd)
        dx2 = dZ * np.ones_like(W_frd)

        return dx1, dx2

    def backward_pass(self, dLo: np.ndarray, input: np.ndarray, diff_s: np.ndarray) -> float:
        # dA ~ derivative of losses
        #dZ = self.activation_func.backward(dA)
        W_frd = self.W_frd
        U_frd = self.U_frd

        # ht_activated
        activated_sum_frd = self.activated_sum

        dV, dsv = self.multiplication_backward(self.output_weights, activated_sum_frd, dLo)

        ds = dsv + diff_s

        dadd = self.activation_func.backward(ds)

        dmulw, dmulu = self.add_backward(U_frd, W_frd, dadd)

        dW, dprev_s = self.multiplication_backward(self.internal_weights, self.activated_sum_prev_layer, dmulw)
        dU, dx = self.multiplication_backward(self.input_weights, input, dmulu)

        # Store gradients weights updates
        self.dprev_s = dprev_s

        self.dU = dU
        self.dW = dW
        self.dV = dV

        # TODO: which values to use for dLo and input? Use dprev_s as diff_s
        return self.previous_layer.backward_pass(dLo, input, dprev_s)
