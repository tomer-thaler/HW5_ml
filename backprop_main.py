import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *
import matplotlib.pyplot as plt

#Name: Tomer Thaler

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


# Section (b): learning-rate sweep
def lr_sweep_section_b():
    """Runs the five-learning-rate experiment and saves three PNG plots."""
    # Loading Data
    np.random.seed(0)
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(10000, 5000)

    epochs = 30
    batch_size = 10
    learning_rates = [1e-3, 1e-2, 1e-1, 1, 10]
    layer_dims = [784, 40, 10]

    # Training loop
    histories = {}
    for lr in learning_rates:
        print(f"\n=== Training with learning rate {lr} ===")
        np.random.seed(0)
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


# section c: training and evaluating on whole mnist
def test_full_section_c():
    """testing test accuracy on the full test size after training on the whole test set"""
    np.random.seed(0)
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(60000, 10000)

    epochs = 30
    batch_size = 10
    layer_dims = [784, 40, 10]
    net = Network(layer_dims)

    #training
    (_, _, _, _, test_acc) = net.train(
        x_train, y_train,
        epochs=epochs, batch_size=batch_size, learning_rate=0.1,
        x_test=x_test, y_test=y_test)
    print(f"Test accuracy on the full test size after {epochs} epochs: {test_acc[-1]}")

# section d: training and evaluating Linear Classifier
def linear_classifier_section_d():
    # Loading Data
    np.random.seed(0)
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(60000, 10000)

    epochs = 30
    batch_size = 10
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

    #Weight visualisation
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

# section e: trying to achieve high acc
def my_NN_section_e():
    """testing test accuracy on the full test size after training on the whole test set, using my setup
    of NN in order to get test accuracy on the whole test set>0.97"""
    np.random.seed(0)
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(60000, 10000)

    epochs = 30
    batch_size = 10
    layer_dims = [784, 128, 128, 10]
    net = Network(layer_dims)

    #training
    learning_rate=0.1
    (_, _, _, _, test_acc) = net.train(
        x_train, y_train,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        x_test=x_test, y_test=y_test)
    print(f"Test accuracy on the full test size after {epochs},"
          f" using {layer_dims} as the structure of the NN, and {learning_rate} as the learning rate: {test_acc[-1]}")


if __name__ == "__main__":
    print(f"today we will explore full NN on mnist, uncomment each section fucntion to see\n")
    #lr_sweep_section_b()   # generates plots for section b
    #test_full_section_c()   # printing test acc for the full mnist set for section c
    #linear_classifier_section_d() # checks how the LC performs here and illustrates it for section d
    #my_NN_section_e()  # shows two inner layers sized 128 gives test acc > 0.97 for section e