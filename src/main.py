import numpy as np
from data_generator.generator import Generator
from enums import DatasetType
from neural_network.layers.dense_layer import DenseLayer
from neural_network.layers.input_layer import InputLayer
from neural_network.layers.recurrent_layer import RecurrentLayer
from neural_network.recurrent_network import RecurrentNetwork
from utils.config_parser import Config
from utils.instantiate import instantiate_activation, instantiate_loss
from utils.load_dataset import load_dataset
from utils.visualize_loss import visualize_loss


def main():
    if Config.generate_data:
        Generator.generate_dataset(Config.shifting_rules, Config.vector_length, Config.sequence_length, Config.dataset_size, Config.bit_probability, Config.split_ratios)

    x_train, y_train, training_rules = load_dataset(DatasetType.TRAINING)
    x_val, y_val, validation_rules = load_dataset(DatasetType.VALIDATION)
    x_test, y_test, test_rules = load_dataset(DatasetType.TEST)

    if Config.train_network:
        # Load network configuration
        layer_neurons = Config.layer_neurons
        layer_types = Config.layer_types
        activation_functions = Config.activation_functions
        weight_ranges = Config.weight_ranges
        custom_learing_rates = Config.custom_learing_rates

        # Configuration validation
        network_lists = [layer_neurons, activation_functions, weight_ranges, custom_learing_rates]
        if any(len(lis) != len(network_lists[0]) for lis in network_lists):
            raise Exception('Invalid network parameter configuration!')

        # Create network, and add input layer
        loss_function = instantiate_loss(Config.loss_function)
        network = RecurrentNetwork(loss_function, Config.learning_rate)
        layer = InputLayer(Config.vector_length)
        network.add_layer(layer)

        # Add hidden layers
        for index in range(0, len(layer_neurons)):
            activation_function = instantiate_activation(activation_functions[index])
            learning_rate = custom_learing_rates[index] if custom_learing_rates[index] else Config.learning_rate
            weight_range = weight_ranges[index] if weight_ranges[index] else Config.weight_ranges

            if layer_types[index] == 0:
                layer = DenseLayer(layer_neurons[index], layer, activation_function, learning_rate, None, weight_range)
            elif layer_types[index] == 1:
                layer = RecurrentLayer(layer_neurons[index], layer, activation_function, learning_rate, weight_range)

            network.add_layer(layer)

        training_losses, validation_losses_tuple = network.fit(x_train, y_train, x_val, y_val, Config.epochs, Config.batch_size)

        visualize_loss(training_losses, validation_losses_tuple)
        test_samples = x_test.shape[0]
        # Test trained model
        x_test = np.transpose(x_test, (1, 0, 2))
        y_test = np.transpose(y_test, (1, 0, 2))

        correct_predictions = 0
        for x in range(test_samples):
            x_test_sample = x_test[:, x, :]
            y_test_sample = y_test[:, x, :]
            test_sample_rule = test_rules[x]

            prediction, loss = network.predict(x_test_sample, y_test_sample)

            # for vector in prediction:
            prediction = np.around(prediction).astype(int)
            print(f'Prediction:\t{prediction[-1]}')
            print(f'Target:\t\t{y_test_sample[-1]}')
            print(test_sample_rule)

            correct_predictions += 1 if np.array_equal(prediction[-1], y_test_sample[-1]) else 0

        print(f'Correct predictions (last sequence): {correct_predictions}/{test_samples} = {round(correct_predictions/test_samples*100, 2)}%')


if __name__ == "__main__":
    main()
