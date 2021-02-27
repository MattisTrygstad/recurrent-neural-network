
import sys
from turtle import forward
import numpy as np
from abstract_classes.layer import Layer
from abstract_classes.loss_function import LossFunction
from neural_network.layers.dense_layer import DenseLayer
from neural_network.layers.recurrent_layer import RecurrentLayer
from neural_network.loss_functions.mean_squared_error import MeanSquaredError
from utils.config_parser import Config
from utils.progress import print_progress


class RecurrentNetwork:

    def __init__(self, loss_function: LossFunction, global_learning_rate: float) -> None:
        self.layers = []
        self.loss_function = loss_function
        self.global_learning_rate = global_learning_rate

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def predict(self, x: np.ndarray, y: np.ndarray = None, add_biases: bool = True) -> tuple:

        final_layer: Layer = self.layers[-1]
        # TODO: support batch size with A and loss calc using for loop
        A = final_layer.forward_pass(x, add_biases)
        if y is not None:
            loss = self.loss_function.compute_loss(A, y)

            if Config.verbose_mode:
                print(x)
                print(A)
                print(loss)
            return A, loss
        else:
            return A, None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, epochs: int, batch_size: int = 64) -> tuple:

        final_layer: Layer = self.layers[-1]

        epoch_training_losses = []
        batch_training_losses = []
        validation_losses_y = []
        validation_losses_x = []

        minibatch_counter = 0
        for epoch in range(epochs):
            epoch_loss = 0

            total_training_samples = x_train.shape[0]
            for sample_num in range(0, total_training_samples, batch_size):
                remaining_samples = total_training_samples - sample_num
                current_batch_size = min(remaining_samples, batch_size)

                x_train_batch = np.transpose(x_train[sample_num:sample_num + current_batch_size], (1, 0, 2))
                y_train_batch = np.transpose(y_train[sample_num:sample_num + current_batch_size], (1, 0, 2))

                # Iterate through sequence length
                seq_length = x_train_batch.shape[0]
                A_seq_array = np.ndarray(x_train_batch.shape)
                for seq_index in range(seq_length):
                    # Make prediction using forward propagation
                    A, loss = self.predict(x_train_batch[seq_index], y_train_batch[seq_index])
                    A_seq_array[seq_index] = np.transpose(A)
                    epoch_loss += loss / seq_length

                dLo_seq_array = np.ndarray(x_train_batch.shape)
                for seq_index in range(seq_length):
                    # Adjust weights and biases using backward propagation
                    # dA ~ loss derivative
                    dLo = self.loss_function.compute_loss_derivative(A_seq_array[seq_index], y_train_batch[seq_index])

                    dLo_seq_array[seq_index] = dLo

                    diff_s = np.zeros((final_layer.output_shape, 1))

                    final_dprev_s = final_layer.backward_pass(dLo, x_train_batch[seq_index], diff_s)

                sys.exit()
                batch_training_losses.append(round(loss / current_batch_size, 10))

                minibatch_counter += 1
                print_progress(sample_num, total_training_samples, length=20)

            epoch_training_losses.append(round(epoch_loss / total_training_samples, 10))

            if x_val.any() and y_val.any():

                A, validation_loss = self.predict(x_val, y_val)

                # TODO: Find solution for this issue
                if Config.loss_function == 0:
                    validation_loss *= 16

                validation_loss = round(validation_loss / x_val.shape[0], 10)
                validation_losses_y.append(validation_loss)
                validation_losses_x.append(minibatch_counter)

                print(f'Epoch: {epoch+1}/{epochs} - Validation loss: {validation_loss} - Training loss: {epoch_training_losses[epoch]}')

        return batch_training_losses, (validation_losses_x, validation_losses_y)
