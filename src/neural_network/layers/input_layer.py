from abstract_classes.layer import Layer
import numpy as np


class InputLayer(Layer):

    def __init__(self, output_shape: int, name: str = 'input') -> None:
        super().__init__(output_shape)
        self.previous_layer = None
        self.name = name

    def forward_pass(self, input: np.ndarray, use_biases: bool) -> None:
        #print(f'input layer forward output shape: {input.shape}')
        # print(input)
        return input

    def backward_pass(self, input: np.ndarray) -> None:
        return 0
