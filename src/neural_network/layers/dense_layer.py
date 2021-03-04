import sys
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
        weighted_sum: np.ndarray = np.transpose(self.weights) @ self.A_previous_layer

        # Add biases
        if add_biases:
            repeats = weighted_sum.shape[-1]
            new_biases = np.repeat(self.biases, repeats, axis=-1)
            weighted_sum += new_biases

        self.Z = weighted_sum
        # Apply activation function
        A = self.activation_func.forward(self.Z)
        #print(f'dense forward output shape: {A.shape}')
        return A

    def backward_pass(self, dLo: np.ndarray, input: np.ndarray, target: np.ndarray, loss_function: LossFunction) -> float:
        # TODO: implement
        return self.previous_layer.backward_pass(dLo, input, target, loss_function)

        """
                # Layer O
        print(target.shape)
        print(V_frd.shape)

        output_jacobian = loss_function.compute_loss_derivative(V_frd, target)

        print(output_jacobian.shape)
        print()

        V_grads = []
        neighbor_jacobians = []

        for x in range(output_jacobian.shape[0]):
            diag = np.diag(output_jacobian[x])
            # print((1 - V_frd[x]**2).shape)
            # print(np.transpose(W_frd)[0].shape)
            # print()
            # print(diag.shape)
            # print(np.outer(1 - V_frd[x]**2, np.transpose(W_frd)[0]).shape)
            V_grad = diag @ np.outer(1 - V_frd[x]**2, np.transpose(W_frd)[0])

            # print(V_grad.shape)
            V_grads.append(V_grad)

            print(np.diag(1 - V_frd[x]).shape)
            print(self.output_weights.shape)
            neighbor_jacobian = np.diag(1 - V_frd[x]**2) @ self.output_weights
            print(neighbor_jacobian)
            sys.exit()

        V_grads = np.array(V_grads)

        print(V_grads.shape)


        """
