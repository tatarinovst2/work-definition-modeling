from pathlib import Path

try:
    import torch  # pylint: disable=import-error
except ImportError:
    torch = None  # type: ignore

from constants import ROOT_DIR


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
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
