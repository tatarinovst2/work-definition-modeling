"""A script for supporting functions"""
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metric(metric: str, log_history: list[dict], output_path: str | Path,
                plot_epochs: bool = True) -> None:
    """
    Plots the metric using information from the log history.
    :param metric: str - the metric to plot (e.g. "rouge-1").
    :param log_history: list[dict] - the log history from the trainer.
    :param output_path: str | Path - the path to save the plot to.
    :param plot_epochs: bool - whether to plot epochs or steps on the x-axis.
    :return: None
    """
    metric_values = []
    steps = []
    epochs = []

    for entry in log_history:
        if metric in entry:
            metric_values.append(entry[metric])
            steps.append(entry['step'])
            epochs.append(entry['epoch'])

    plt.figure(figsize=(10, 5))

    if plot_epochs:
        plt.plot(epochs, metric_values, label=metric, marker='o', linestyle='-', color='b')
        plt.title(f"{metric} Over Epochs")
        plt.xlabel('Epochs')
    else:
        plt.plot(steps, metric_values, label=metric, marker='o', linestyle='-', color='b')
        plt.title(f"{metric} Over Steps")
        plt.xlabel('Steps')

    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)


def plot_training_and_test_loss(log_history: list[dict], output_path: str | Path,
                                plot_epochs: bool = True) -> None:
    """
    Plots the training and test loss using information from the log history.
    :param log_history: list[dict] - the log history from the trainer.
    :param output_path: str | Path - the path to save the plot to.
    :param plot_epochs: bool - whether to plot epochs or steps on the x-axis.
    :return: None
    """
    train_losses = []
    test_losses = []
    steps = []
    epochs = []

    for entry in log_history:
        if 'loss' in entry:
            train_losses.append(entry['loss'])
            steps.append(entry['step'])
            epochs.append(entry['epoch'])
        if 'eval_loss' in entry:
            test_losses.append(entry['eval_loss'])

    if len(train_losses) != len(test_losses):
        print(f"Train losses: {train_losses}, test losses: {test_losses}, "
              f"steps: {steps}, epochs: {epochs}")
        raise ValueError("Train losses and test losses have different lengths")

    plt.figure(figsize=(10, 5))

    if plot_epochs:
        plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-',
                 color='b')
        plt.plot(epochs, test_losses, label='Test Loss', marker='o', linestyle='-', color='r')
        plt.title('Training and Test Loss Over Epochs')
        plt.xlabel('Epochs')
    else:
        plt.plot(steps, train_losses, label='Training Loss', marker='o', linestyle='-',
                    color='b')
        plt.plot(steps, test_losses, label='Test Loss', marker='o', linestyle='-', color='r')
        plt.title('Training and Test Loss Over Steps')
        plt.xlabel('Steps')

    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
