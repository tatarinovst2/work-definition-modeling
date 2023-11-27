"""A module which splits the raw dataset into train, test and validation splits."""
import argparse

from src.dataset_processing import prepare_dataset
from src.utils import parse_path


def main() -> None:
    """Split the raw dataset."""
    parser = argparse.ArgumentParser(
        description="Split the raw dataset into train, test and validation splits.")

    parser.add_argument("raw_dataset_path",
                        type=str,
                        help="The path to the raw dataset.")
    parser.add_argument("--train-split", "-t",
                        type=float,
                        default=0.8,
                        help="The proportion of the dataset to use for training.")
    parser.add_argument("--output-dir", "-o",
                        type=str,
                        default="data/splits",
                        help="The directory to save the splits to.")

    args = parser.parse_args()

    parsed_raw_dataset_path = parse_path(args.raw_dataset_path)
    parsed_output_dir = parse_path(args.output_dir)

    prepare_dataset(parsed_raw_dataset_path, parsed_output_dir, train_size=args.train_split)


if __name__ == "__main__":
    main()
