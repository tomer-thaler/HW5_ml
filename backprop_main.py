import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *
import matplotlib.pyplot as plt

def plot_metric(histories, lr_list, metric_index, ylabel,
                filename, log=False):
    plt.figure()
    for lr in lr_list:
        metric = histories[lr][metric_index]
        plt.plot(range(1, len(metric) + 1), metric, label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    if log:
        plt.yscale('log')
    plt.title(f"{ylabel} vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")


# ------------------------------------------------------------
# Section (b): learning-rate sweep wrapped in one function
# ------------------------------------------------------------
def lr_sweep_section_b():
    """Runs the five-learning-rate experiment and saves three PNG plots."""
    # Loading Data
    np.random.seed(0)
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(10000, 5000)

    # Training configuration
    epochs       = 30
    batch_size   = 10
    learning_rates = [1e-3, 1e-2, 1e-1, 1, 10]

    # Network configuration
    layer_dims = [784, 40, 10]

    # Training loop
    histories = {}
    for lr in learning_rates:
        print(f"\n=== Training with learning rate {lr} ===")
        np.random.seed(0)                       # same init each run
        net = Network(layer_dims)
        (_, train_loss, _, train_acc, test_acc) = net.train(
                x_train, y_train,
                epochs=epochs, batch_size=batch_size, learning_rate=lr,
                x_test=x_test, y_test=y_test)
        histories[lr] = (train_loss, train_acc, test_acc)

    # Plotting
    plot_metric(histories, learning_rates, 0, "Training loss",
                "b_train_loss.png", log=True)
    plot_metric(histories, learning_rates, 1, "Training accuracy",
                "b_train_acc.png")
    plot_metric(histories, learning_rates, 2, "Test accuracy",
                "b_test_acc.png")



def test_full_section_c():
    """testing test accuracy on the full test size after training on the whole test set"""
    np.random.seed(0)
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(60000, 10000)

    # Training configuration
    epochs = 30
    batch_size = 10

    # Network configuration
    layer_dims = [784, 40, 10]
    net = Network(layer_dims)

    #training
    (_, _, _, _, test_acc) = net.train(
        x_train, y_train,
        epochs=epochs, batch_size=batch_size, learning_rate=0.1,
        x_test=x_test, y_test=y_test)
    print(f"Test accuracy on the full test size after {epochs} epochs: {test_acc[-1]}")

def linear_classifier_section_d():
    # Loading Data
    np.random.seed(0)
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(60000, 10000)

    # Training configuration
    epochs = 30
    batch_size = 10

    # Network configuration
    layer_dims = [784, 10]

    # Training

    net = Network(layer_dims)
    (_, _, _, train_acc, test_acc) = net.train(
        x_train, y_train,
        epochs=epochs, batch_size=batch_size, learning_rate=0.1,
        x_test=x_test, y_test=y_test)

    # Plotting the test & train accuracy over epochs
    plt.figure()
    plt.plot(range(1, epochs + 1), train_acc, label="Train")
    plt.plot(range(1, epochs + 1), test_acc, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Linear classifier – accuracy vs epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("d_linear_accuracy.png")
    plt.close()
    print("Saved d_linear_accuracy.png")

    #Weight visualisation (10 templates)
    W = net.parameters["W1"]  # shape (10, 784)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.ravel()
    for i in range(10):
        axes[i].imshow(W[i].reshape(28, 28),interpolation='nearest')
        axes[i].set_title(f"Digit {i}")
        axes[i].axis("off")

    fig.suptitle("Linear-classifier weight templates")
    plt.tight_layout()
    plt.savefig("d_linear_weights.png")
    plt.close()
    print("Saved d_linear_weights.png")


if __name__ == "__main__":
    #lr_sweep_section_b()   # generates plots for part (b)
    #test_full_section_c()   # printing test acc for the full mnist set
    linear_classifier_section_d()