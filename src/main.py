
import random
import sys
from matplotlib import pyplot as plt
import numpy as np
from data_generator.generator import Generator
from enums import DatasetType
from neural_network.activation_functions.relu import Relu
from neural_network.activation_functions.sigmoid import Sigmoid
from neural_network.activation_functions.softmax import Softmax
from neural_network.layers.dense_layer import DenseLayer
from neural_network.layers.input_layer import InputLayer
from neural_network.loss_functions.cross_entropy import CrossEntropy
from neural_network.loss_functions.mean_squared_error import MeanSquaredError
from neural_network.network import Network
from utils.config_parser import Config
from utils.instantiate import instantiate_activation, instantiate_loss, instantiate_regularizer
from utils.load_dataset import load_dataset
from utils.visualize_image import visualize_image
from utils.visualize_loss import visualize_loss


def main():
    # Generate training, validataion and test data
    if Config.generate_data:
        generator = Generator()
        generator.generate_dataset(Config.dataset_size, canvas_size=Config.canvas_size, noise_ratio=Config.noise_ratio, pos_deviaiton=Config.pos_deviaiton, line_width_deviaiton=Config.line_width_deviaiton, width_range=Config.width_range, height_range=Config.height_range, split_ratios=Config.split_ratios)

    # Load training data
    x_train, y_train = load_dataset(DatasetType.TRAINING, True)
    y_train = np.array([[1 if x == y_train[n] else 0 for x in range(4)] for n in range(y_train.size)])

    # Load validation data
    x_val, y_val = load_dataset(DatasetType.VALIDATION, True)
    y_val = np.array([[1 if x == y_val[n] else 0 for x in range(4)] for n in range(y_val.size)])

    # Load test data
    x_test, y_test = load_dataset(DatasetType.TEST, True)
    y_test = np.array([[1 if x == y_test[n] else 0 for x in range(4)] for n in range(y_test.size)])

    if Config.train_network:
        # Load network configuration
        hidden_layers = Config.hidden_layers
        activation_functions = Config.activation_functions
        weight_ranges = Config.weight_ranges
        custom_learing_rates = Config.custom_learing_rates
        regularizer = instantiate_regularizer(Config.regularizer, Config.regularizer_rate)

        # Configuration validation
        network_lists = [hidden_layers, activation_functions, weight_ranges, custom_learing_rates]
        if any(len(lis) != len(network_lists[0]) for lis in network_lists):
            raise Exception('Invalid network parameter configuration!')

        # Create network, and add input layer
        loss_function = instantiate_loss(Config.loss_function)
        network = Network(loss_function, Config.learning_rate)
        layer = InputLayer(x_train.shape[1])
        network.add_layer(layer)

        # Add hidden layers
        for index in range(len(hidden_layers)):
            activation_function = instantiate_activation(activation_functions[index])
            learning_rate = custom_learing_rates[index] if custom_learing_rates[index] else Config.learning_rate
            weight_range = weight_ranges[index] if weight_ranges[index] else Config.weight_ranges

            layer = DenseLayer(hidden_layers[index], layer, activation_function, learning_rate, regularizer, weight_range)
            network.add_layer(layer)

        # Output layer
        activation_function = instantiate_activation(Config.output_activation_function)
        output = DenseLayer(4, layer, activation_function, Config.learning_rate, None)
        network.add_layer(output)

        training_losses, validation_losses_tuple = network.fit(x_train, y_train, x_val, y_val, Config.epochs)

        visualize_loss(training_losses, validation_losses_tuple)

    num_test_samples = x_test.shape[0]
    for x in range(Config.display_images):

        random_index = random.randint(0, num_test_samples)

        prediction = None
        if Config.train_network:
            # Make single prediction on sample from validation data
            prediction, loss = network.predict(x_val[random_index], None, False)

        # Answer
        target = y_val[random_index]

        visualize_image(x_val[random_index], prediction, target)


if __name__ == "__main__":
    main()
