import copy
import math
from matplotlib import figure, pyplot as plt
from matplotlib import colors
import numpy as np

from enums import ImageType


def visualize_image(grid_data: np.ndarray, prediction: np.ndarray = None, target: np.ndarray = None) -> None:
    dpi = 96.0
    fig: figure.Figure = plt.figure(figsize=(8, 8))
    #im = plt.imshow(X=grid_data, cmap=plt.cm.binary, origin='lower', interpolation='none', vmin=0, vmax=1, aspect='equal')
    cmap = colors.ListedColormap(['white', 'black'])
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    if len(grid_data.shape) == 1:
        pixels = grid_data.shape[0]
        size = int(math.sqrt(pixels))
        grid_data = np.reshape(grid_data, (size, size))

    plt.pcolormesh(grid_data, cmap=cmap, norm=norm, edgecolors='grey', linewidth=dpi / (1024 * 32))

    if prediction is not None:
        prediction = [round(value, 2) for value in prediction]
        prediction_index = prediction.index(max(prediction))
        target_index = list(target).index(max(target))

        shape = ImageType(prediction_index).name
        if prediction_index == target_index:
            result = f'The network predicted {shape.lower()}, which is correct!'
            color = 'green'
        else:
            result = 'The prediction was wrong..'
            color = 'red'

        fig.suptitle(f'Prediction: {str(prediction)}\nTarget: {str(target)}', fontsize=14)
        plt.figtext(0.5, 0.03, f'{result}', ha='center', fontsize=18, color=color)
    plt.show()
