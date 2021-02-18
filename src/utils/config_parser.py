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

    # NETWORK
    epochs = int(ast.literal_eval(config.get('NETWORK', 'epochs')))
    loss_function = int(ast.literal_eval(config.get('NETWORK', 'loss_function')))
    regularizer = int(ast.literal_eval(config.get('NETWORK', 'regularizer')))
    regularizer_rate = float(ast.literal_eval(config.get('NETWORK', 'regularizer_rate')))
    learning_rate = float(ast.literal_eval(config.get('NETWORK', 'learning_rate')))

    # LAYERS
    hidden_layers = list(ast.literal_eval(config.get('LAYERS', 'hidden_layers')))
    activation_functions = list(ast.literal_eval(config.get('LAYERS', 'activation_functions')))
    weight_ranges = list(ast.literal_eval(config.get('LAYERS', 'weight_ranges')))
    custom_learing_rates = list(ast.literal_eval(config.get('LAYERS', 'custom_learing_rates')))
    output_activation_function = int(ast.literal_eval(config.get('LAYERS', 'output_activation_function')))

    # DATA_GENERATION
    dataset_size = int(ast.literal_eval(config.get('DATA_GENERATION', 'dataset_size')))
    canvas_size = int(ast.literal_eval(config.get('DATA_GENERATION', 'canvas_size')))
    noise_ratio = float(ast.literal_eval(config.get('DATA_GENERATION', 'noise_ratio')))
    pos_deviaiton = float(ast.literal_eval(config.get('DATA_GENERATION', 'pos_deviaiton')))
    line_width_deviaiton = float(ast.literal_eval(config.get('DATA_GENERATION', 'line_width_deviaiton')))
    width_range = tuple(ast.literal_eval(config.get('DATA_GENERATION', 'width_range')))
    height_range = tuple(ast.literal_eval(config.get('DATA_GENERATION', 'height_range')))
    split_ratios = tuple(ast.literal_eval(config.get('DATA_GENERATION', 'split_ratios')))
