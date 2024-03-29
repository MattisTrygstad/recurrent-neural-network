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
        Generator.generate_dataset(Config.shifting_rules, Config.vector_length, Config.sequence_length, Config.dataset_size, Config.bit_ratio, Config.split_ratios)

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

        # Test trained model
        test_samples = x_test.shape[0]
        correct_predictions = 0

        for x in range(test_samples):
            x_test_sample = np.transpose(x_test[x:x + 1], (1, 0, 2))
            y_test_sample = np.transpose(y_test[x:x + 1], (1, 0, 2))

            test_sample_rule = test_rules[x]

            predictions = []
            for seq_index in range(x_test_sample.shape[0]):
                pred, _ = network.predict(x_test_sample[seq_index])
                predictions.append(pred)

            # for vector in prediction:
            prediction = np.where(predictions[-1] > 0.5, 1, 0).astype(int)
            print(f'Prediction:\t{prediction[0]}')
            print(f'Target:\t\t{y_test_sample[-1][0]}')
            print(test_sample_rule)
            print()

            correct_predictions += 1 if np.array_equal(prediction[0], y_test_sample[-1][0]) else 0
            network.reset_memory()

        accuracy = round(correct_predictions / test_samples * 100, 2)
        print(f'Correct predictions: {correct_predictions}/{test_samples} = {accuracy}%')

        visualize_loss(training_losses, validation_losses_tuple, accuracy)


if __name__ == "__main__":
    main()
