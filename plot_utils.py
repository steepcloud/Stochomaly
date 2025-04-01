import matplotlib.pyplot as plt
import os

def plot_loss(loss_history, optimizer, activation, save_path="plots/training_loss.png"):
    """Plots and saves the training loss graph."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()
    plt.plot(range(len(loss_history)), loss_history, label=f"{optimizer} - {activation}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.show()
