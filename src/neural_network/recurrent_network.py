
from xxlimited import Null
import numpy as np
from abstract_classes.layer import Layer
from abstract_classes.loss_function import LossFunction
from neural_network.layers.dense_layer import DenseLayer
from neural_network.layers.recurrent_layer import RecurrentLayer
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
        prediction = final_layer.forward_pass(x, add_biases)
        if y is not None:
            loss = self.loss_function.compute_loss(prediction, y)

            if Config.verbose_mode:
                print(f'Network input:\n{x}')
                print(f'Prediction:\n{prediction}')
                print(f'Target:\n{y}')
                print(f'Loss:\n{loss}')
            return prediction, loss
        else:
            return prediction, None

    def reset_memory(self):
        for layer in self.layers:
            if isinstance(layer, RecurrentLayer):
                layer.activated_sums_prev_layer = []
                layer.activated_sums = []
                if hasattr(layer, 'delta_jacobian_cumulative'):
                    del layer.delta_jacobian_cumulative
                if hasattr(layer, 'U_grad_cumulative'):
                    del layer.U_grad_cumulative
                if hasattr(layer, 'W_grad_cumulative'):
                    del layer.W_grad_cumulative

            if isinstance(layer, DenseLayer):
                layer.activated_sums = []
                layer.activated_sums_prev_layer = []
                if hasattr(layer, 'V_grad_cumulative'):
                    del layer.V_grad_cumulative

    def update_weights(self):
        for layer in self.layers:
            if isinstance(layer, RecurrentLayer):
                layer.input_weights -= layer.learning_rate * layer.U_grad_cumulative
                layer.internal_weights -= layer.learning_rate * layer.W_grad_cumulative

            if isinstance(layer, DenseLayer):
                layer.weights -= layer.learning_rate * layer.V_grad_cumulative

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
                minibatch_counter += 1
                remaining_samples = total_training_samples - sample_num
                current_batch_size = min(remaining_samples, batch_size)

                x_train_batch = np.transpose(x_train[sample_num:sample_num + current_batch_size], (1, 0, 2))
                y_train_batch = np.transpose(y_train[sample_num:sample_num + current_batch_size], (1, 0, 2))

                seq_length = x_train_batch.shape[0]
                activated_sum_seq_array = []

                seq_losses = []
                for seq_index in range(seq_length):
                    # Make prediction using forward propagation
                    prediction, batch_losses = self.predict(x_train_batch[seq_index], y_train_batch[seq_index])

                    activated_sum_seq_array.append(prediction)
                    seq_losses.append(batch_losses)

                epoch_loss += seq_losses[-1]
                batch_training_losses.append(seq_losses[-1])

                dLo_seq_array = []
                for seq_index in range(seq_length - 1, -1, -1):
                    # Loss derivative
                    dLo = self.loss_function.compute_loss_derivative(activated_sum_seq_array[seq_index], y_train_batch[seq_index])
                    dLo_seq_array.append(dLo)

                    # Backpropagation through the network
                    output_jacobian = final_layer.backward_pass(dLo)

                self.update_weights()
                self.reset_memory()

                print_progress(sample_num, total_training_samples, length=20)

            epoch_training_losses.append(round(epoch_loss / total_training_samples, 10))

            if x_val.any() and y_val.any():
                total_validation_samples = x_val.shape[0]
                val_batch_loss = 0

                counter = 0
                for sample_num in range(0, total_validation_samples, batch_size):
                    counter += 1

                    remaining_samples = total_validation_samples - sample_num
                    current_batch_size = min(remaining_samples, batch_size)

                    x_val_batch = np.transpose(x_val[sample_num:sample_num + current_batch_size], (1, 0, 2))
                    y_val_batch = np.transpose(y_val[sample_num:sample_num + current_batch_size], (1, 0, 2))

                    seq_losses = []
                    for seq_index in range(seq_length):
                        # Make prediction using forward propagation
                        prediction, val_loss = self.predict(x_val_batch[seq_index], y_val_batch[seq_index])

                        seq_losses.append(val_loss)

                    val_batch_loss += seq_losses[-1]
                    self.reset_memory()

                total_val_loss = round(val_batch_loss / counter, 10)
                validation_losses_y.append(total_val_loss)
                validation_losses_x.append(minibatch_counter)

                print(f'Epoch: {epoch+1}/{epochs} - Validation loss: {total_val_loss} - Training loss: {epoch_training_losses[epoch]}')

        return batch_training_losses, (validation_losses_x, validation_losses_y)
