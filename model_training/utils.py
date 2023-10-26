"""A script for supporting functions"""
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_train_config(config_path: str | Path) -> dict:
    """
    Loads a config from a JSON file.
    :param config_path: str | Path - the path to the config.
    :return: dict - the config.
    """
    with open(config_path, "r", encoding="utf-8") as json_file:
        config = json.load(json_file)

    required_keys = ["model_checkpoint", "dataset_path", "batch_size"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config must contain the key '{key}'")

    if "max_steps" not in config and "num_train_epochs" not in config:
        raise ValueError("Config must contain either 'max_steps' or 'num_train_epochs'")

    if "max_steps" in config and "num_train_epochs" in config:
        raise ValueError("Config must not contain both 'max_steps' and 'num_train_epochs'")

    if "save_steps" in config and config.get("save_strategy", "epoch") != "steps":
        raise ValueError("Config must not contain 'save_steps' and 'save_strategy' "
                         "other than 'steps'")

    if "logging_steps" in config and config.get("logging_strategy", "epoch") != "steps":
        raise ValueError("Config must not contain 'logging_steps' and 'logging_strategy' "
                         "other than 'steps'")

    if "eval_steps" in config and config.get("evaluation_strategy", "epoch") != "steps":
        raise ValueError("Config must not contain 'eval_steps' and 'evaluation_strategy' "
                         "other than 'steps'")

    return config


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
