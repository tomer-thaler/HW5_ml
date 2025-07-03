import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *
import matplotlib.pyplot as plt

def plot_metric(metric_index, ylabel, filename, log=False):
    """plot generating function showing train loss, train acc, test acc, with different learning rates over epochs
        """
    plt.figure()
    for lr in learning_rates:
        metric = histories[lr][metric_index]
        plt.plot(range(1, epochs + 1), metric, label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    if log:
        plt.yscale('log')
    plt.title(f"{ylabel} v.s. Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

# Loading Data
np.random.seed(0)  # For reproducibility
n_train = 10000
n_test = 5000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

# Training configuration
epochs = 30
batch_size = 10
learning_rates = [1e-3, 1e-2, 1e-1, 1, 10] #for section b


# Network configuration
layer_dims = [784, 40, 10]

# training + getting train loss, train acc, test acc, with different learning rates
histories = {}
for lr in learning_rates:
    print(f"\n=== Training with learning rate {lr} ===")
    np.random.seed(0)                 # same weight init & shuffle every run
    net = Network(layer_dims)
    _, train_loss, _, train_acc, test_acc = net.train(x_train, y_train, epochs, batch_size, learning_rate=lr, x_test=x_test, y_test=y_test)
    histories[lr] = (train_loss, train_acc, test_acc)

#plotting the plots for section (b)
plot_metric(metric_index=0, ylabel="Training loss",
            filename="b_train_loss.png", log=True)

plot_metric(metric_index=1, ylabel="Training accuracy",
            filename="b_train_acc.png")

plot_metric(metric_index=2, ylabel="Test accuracy",
            filename="b_test_acc.png")


