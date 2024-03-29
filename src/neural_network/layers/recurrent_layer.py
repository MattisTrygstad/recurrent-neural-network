import numpy as np
from abstract_classes.activation_function import ActivationFunction
from abstract_classes.layer import Layer
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
        self.activated_sums_prev_layer = []
        self.activated_sums = []

        # Layer weights
        self.input_weights = np.random.uniform(low=init_weight_range[0], high=init_weight_range[1], size=(self.input_shape, self.output_shape))
        self.internal_weights = np.random.uniform(low=init_weight_range[0], high=init_weight_range[1], size=(self.output_shape, self.output_shape))

        if name:
            self.name = name
        else:
            self.name = f'recurrent{self.output_shape}'

    def forward_pass(self, input: np.ndarray, add_biases: bool) -> np.ndarray:
        # print(f'forward  {self.name}')
        # print('recurrent forward')
        # Send each case through the network from input to output
        activated_sum_prev_layer = self.previous_layer.forward_pass(input, add_biases)
        activated_sum_prev_seq = np.zeros((1, self.output_shape)) if len(self.activated_sums) == 0 else self.activated_sums[-1]

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

        self.activated_sums_prev_layer.append(activated_sum_prev_layer)
        self.activated_sums.append(activated_sum)

        return activated_sum

    def backward_pass(self, output_jacobian: np.ndarray) -> float:
        # print(f'backward  {self.name}')

        activated_sum_prev_layer = self.activated_sums_prev_layer.pop()

        batch_size = output_jacobian.shape[0]

        # First sequence in backprop
        if not hasattr(self, 'delta_jacobian_cumulative') or not Config.recurrence:
            # Weigh grad params
            W_grad_prev_seq = np.zeros_like(self.internal_weights)
            U_grad_prev_seq = np.zeros_like(self.input_weights)

            # Delta jacobian params
            delta_jacobian = output_jacobian
        else:
            # Weight grad params
            W_grad_prev_seq = self.W_grad_cumulative
            U_grad_prev_seq = self.U_grad_cumulative

            # Delta jacobian params
            # H_k+1
            next_activated_sum = self.activated_sums.pop()
            delta_jacobian_prev_seq = self.delta_jacobian_cumulative

            recurrent_jacobian = [np.diag(self.activation_func.backward(next_activated_sum[x])) @ np.transpose(self.internal_weights) for x in range(batch_size)]
            recurrent_jacobian = np.sum(recurrent_jacobian, axis=0)

            delta_jacobian = output_jacobian + delta_jacobian_prev_seq @ recurrent_jacobian

        self.delta_jacobian_cumulative = delta_jacobian

        # H_k
        curr_activated_sum = self.activated_sums[-1]
        # H_k-1
        prev_activated_sum = np.zeros_like(output_jacobian) if len(self.activated_sums) == 1 else self.activated_sums[-2]

        # print('delta_jacobian', delta_jacobian.shape)

        # Shapes: W_grad = W_grad_prev_seq = (recurrent_size, recurrent_size), output_jacobian = (batch_size, bit_vector_size), curr_activated_sum = prev_activated_sum = (batch_size, recurrent_size)
        self.W_grad_cumulative = self.compute_weight_gradient(delta_jacobian, self.activation_func, curr_activated_sum, prev_activated_sum, batch_size, W_grad_prev_seq)

        # print('W_grad', W_grad.shape)

        # TODO: Add shapes
        self.U_grad_cumulative = self.compute_weight_gradient(delta_jacobian, self.activation_func, curr_activated_sum, activated_sum_prev_layer, batch_size, U_grad_prev_seq)
        # print('U_grad', U_grad.shape)

        neighbor_jacobian = self.compute_neighbor_jacobian(self.activation_func, curr_activated_sum, self.input_weights, batch_size)

        # print('neighbor_jacobian', neighbor_jacobian.shape)

        next_output_jacobian = delta_jacobian @ neighbor_jacobian

        # print('next_output_jacobian', next_output_jacobian.shape)

        return self.previous_layer.backward_pass(next_output_jacobian)
