

from matplotlib import pyplot as plt


def visualize_loss(training_losses: list, validation_losses: tuple, accuracy: float) -> None:
    plt.plot(training_losses, label='Training loss')
    plt.plot(validation_losses[0], validation_losses[1], label='Validation loss')
    plt.xlabel('Minibatches')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    text = f"Test data accuracy (last sequence): {accuracy}%"
    plt.title(text)
    plt.show()
