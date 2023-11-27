"""A script for supporting functions."""
import json
from pathlib import Path

try:
    import torch  # pylint: disable=import-error
except ImportError:
    torch = None  # type: ignore

from .constants import ROOT_DIR


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

def parse_path(path: str | Path) -> Path:
    """
    Ensure that the path is absolute and is in a pathlib.Path format.

    :param path: The path to parse.
    :return: The parsed path.
    """
    path = Path(path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def get_current_torch_device() -> str:
    """
    Get the current torch device.

    :return: The current torch device.
    """
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"