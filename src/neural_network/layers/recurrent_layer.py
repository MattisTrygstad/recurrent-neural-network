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
        self.init_W = np.zeros((output_shape, 1))
        self.activated_sum_prev_layer = []
        self.activated_sum = []
        self.U_frd = []
        self.W_frd = []
        self.V_frd = []

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
        # Send each case through the network from input to output
        activated_sum_prev_layer = self.previous_layer.forward_pass(input, add_biases)

        W_prev_seq = self.init_W if len(self.W_frd) == 0 else self.W_frd[-1]
        # Multiply the outputs of the previous layer with the weights
        W_frd: np.ndarray = self.internal_weights @ W_prev_seq

        # print('U factors')
        # print(np.transpose(self.input_weights).shape)
        # print(activated_sum_prev_layer.shape)
        U_frd: np.ndarray = np.transpose(self.input_weights) @ activated_sum_prev_layer

        print()
        print(U_frd.shape)
        print(W_frd.shape)

        temp_sum = W_frd + U_frd

        # Add biases
        if add_biases:
            repeats = W_frd.shape[-1]
            new_biases = np.repeat(self.biases, repeats, axis=-1)
            temp_sum += new_biases

        # Apply activation function
        activated_sum = self.activation_func.forward(temp_sum)

        V_frd = self.output_weights @ activated_sum

        # print(f'V{V_frd.shape}')

        self.activated_sum_prev_layer.append(activated_sum_prev_layer)
        self.activated_sum.append(activated_sum)
        self.U_frd.append(U_frd)
        self.W_frd.append(W_frd)
        self.V_frd.append(V_frd)

        return activated_sum

    def multiplication_backward(self, weights: np.ndarray, frd: np.ndarray, grad: np.ndarray):
        gradient_weight = grad @ np.transpose(frd)
        chain_gradient = np.transpose(weights) @ grad

        return gradient_weight, chain_gradient

    def add_backward(self, U_frd: np.ndarray, W_frd: np.ndarray, dZ: np.ndarray):
        dx1 = dZ * np.ones_like(U_frd)
        dx2 = dZ * np.ones_like(W_frd)

        return dx1, dx2

    def backward_pass(self, dLo: np.ndarray, input: np.ndarray, target: np.ndarray, loss_function: LossFunction, output_pred: np.ndarray) -> float:
        # dLo ~ derivative of losses - Output jacobian
        # Recurrent jacobian (?) derivative of output wrt. output prev iteration
        # dU, dW ~ Weight jacobians (input, internal/recurrent) (dV)
        # dmulw, dmulu ~ Delta jacobian

        W_frd = self.W_frd.pop()
        U_frd = self.U_frd.pop()
        V_frd = self.V_frd.pop()
        V_frd_prev_seq = self.V_frd[-1]

        curr_activated_sum = self.activated_sum.pop()
        prev_activated_sum = self.activated_sum[-1]

        activated_sum_prev_layer = self.activated_sum_prev_layer.pop()

        #print(np.diag(np.transpose(1 - W_frd**2)[0]))
        # TODO: ref. forelesning, bruke V her??
        recurrent_jacobian = np.diag(np.transpose(1 - W_frd**2)[0]) @ np.transpose(self.internal_weights)

        # print(recurrent_jacobian.shape)

        print('backprop')
        print(activated_sum_prev_layer.shape)
        print(target.shape)

        loss_derivative = loss_function.compute_loss_derivative(activated_sum_prev_layer, target)
        print(loss_derivative.shape)

        U_grad = [np.diag(loss_function.compute_loss_derivative(V_frd[x], target[x])) @ (np.outer(1 - V_frd[x]**2, input[x])) for x in range(activated_sum_prev_layer.shape[0])]
        U_grad = np.array(U_grad)

        print(U_grad.shape)
        print(self.input_weights.shape)

        V_grad = [np.diag(loss_function.compute_loss_derivative(output_pred[x], target[x])) @ (np.outer(1 - output_pred[x]**2, input[x])) for x in range(output_pred.shape[0])]
        V_grad = np.array(V_grad)
        print(V_grad.shape)
        print(self.output_weights.shape)

        W_grad = [np.diag(loss_function.compute_loss_derivative(V_frd[x], target[x])) @ (np.outer(1 - V_frd[x]**2, V_frd_prev_seq[x])) for x in range(V_frd.shape[0])]
        W_grad = np.array(W_grad)
        print(W_grad.shape)
        print(self.internal_weights.shape)
        sys.exit()
        print(V_frd.shape)
        print(target.shape)
        print(loss_function.compute_loss_derivative(V_frd, target).shape)
        print()
        print(V_frd.shape)
        print(activated_sum_prev_layer.shape)
        print()
        # TODO: add together all timesteps
        # O_k ~ V_frd, H_k ~ activated_sum_prev_layer
        V_grad = [np.diag(loss_function.compute_loss_derivative(V_frd[x], target[x])) @ (np.outer(1 - V_frd[x]**2, activated_sum_prev_layer[x])) for x in range(V_frd.shape[0])]
        V_grad = np.array(V_grad)
        print(V_grad.shape)
        sys.exit()
        ds = diff_s

        dadd = self.activation_func.backward(ds)

        dmulw, dmulu = self.add_backward(U_frd, W_frd, dadd)

        dW, dprev_s = self.multiplication_backward(self.internal_weights, activated_sum_prev_layer, dmulw)
        dU, dx = self.multiplication_backward(self.input_weights, input, dmulu)

        # Store gradients weights updates
        self.dprev_s = dprev_s

        # TODO: store for each sequence
        self.dU = dU
        self.dW = dW

        # TODO: which values to use for dLo and input? Use dprev_s as diff_s
        return self.previous_layer.backward_pass(dLo, input, dprev_s)
