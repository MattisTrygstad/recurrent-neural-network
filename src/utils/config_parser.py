import ast
import sys
import configparser


class Config:
    config = configparser.ConfigParser()
    config.read('config.ini')

    # GLOBAL
    generate_data = bool(ast.literal_eval(config.get('GLOBAL', 'generate_data')))
    train_network = bool(ast.literal_eval(config.get('GLOBAL', 'train_network')))
    display_images = int(ast.literal_eval(config.get('GLOBAL', 'display_images')))
    verbose_mode = bool(ast.literal_eval(config.get('GLOBAL', 'verbose_mode')))
    dataset_name = str(ast.literal_eval(config.get('GLOBAL', 'dataset_name')))

    # NETWORK
    epochs = int(ast.literal_eval(config.get('NETWORK', 'epochs')))
    batch_size = int(ast.literal_eval(config.get('NETWORK', 'batch_size')))
    loss_function = int(ast.literal_eval(config.get('NETWORK', 'loss_function')))
    learning_rate = float(ast.literal_eval(config.get('NETWORK', 'learning_rate')))

    # LAYERS
    input_size = int(ast.literal_eval(config.get('LAYERS', 'input_size')))
    layer_neurons = list(ast.literal_eval(config.get('LAYERS', 'layer_neurons')))
    layer_types = list(ast.literal_eval(config.get('LAYERS', 'layer_types')))
    activation_functions = list(ast.literal_eval(config.get('LAYERS', 'activation_functions')))
    weight_ranges = list(ast.literal_eval(config.get('LAYERS', 'weight_ranges')))
    custom_learing_rates = list(ast.literal_eval(config.get('LAYERS', 'custom_learing_rates')))

    # DATA_GENERATION
    dataset_size = int(ast.literal_eval(config.get('DATA_GENERATION', 'dataset_size')))
    split_ratios = tuple(ast.literal_eval(config.get('DATA_GENERATION', 'split_ratios')))
    shifting_rules = list(ast.literal_eval(config.get('DATA_GENERATION', 'shifting_rules')))
    sequence_length = int(ast.literal_eval(config.get('DATA_GENERATION', 'sequence_length')))
    vector_length = int(ast.literal_eval(config.get('DATA_GENERATION', 'vector_length')))
    bit_probability = float(ast.literal_eval(config.get('DATA_GENERATION', 'bit_probability')))
