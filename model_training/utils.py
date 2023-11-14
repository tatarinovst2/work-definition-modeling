"""A script for supporting functions."""
import json
from pathlib import Path


def load_train_config(config_path: str | Path) -> dict:
    """
    Load a config from a JSON file.

    :param config_path: The path to the config.
    :return: The config.
    :raises ValueError: If the config is invalid.
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
