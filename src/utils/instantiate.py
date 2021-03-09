

from abstract_classes.activation_function import ActivationFunction
from abstract_classes.loss_function import LossFunction
from abstract_classes.regularizer import Regularizer
from neural_network.activation_functions.linear import Linear
from neural_network.activation_functions.relu import Relu
from neural_network.activation_functions.sigmoid import Sigmoid
from neural_network.activation_functions.softmax import Softmax
from neural_network.activation_functions.tanh import Tanh
from neural_network.loss_functions.cross_entropy import CrossEntropy
from neural_network.loss_functions.mean_squared_error import MeanSquaredError


def instantiate_activation(num: int) -> ActivationFunction:
    if num == 0:
        return Sigmoid()
    elif num == 1:
        return Tanh()
    elif num == 2:
        return Relu()
    elif num == 3:
        return Linear()
    elif num == 4:
        return Softmax()
    else:
        raise Exception('Invalid activation function')


def instantiate_loss(num: int) -> LossFunction:
    if num == 0:
        return MeanSquaredError
    elif num == 1:
        return CrossEntropy
    else:
        raise Exception('Invalid loss function')
