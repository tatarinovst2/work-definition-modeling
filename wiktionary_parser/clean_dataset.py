"""A module for cleaning the dataset."""
import argparse
import json
from pathlib import Path


def load_dataset(dataset_path: str | Path) -> list[dict[str, str]]:
    """
    Load the dataset of jsonl format from the given path.

    :param dataset_path: The path to the dataset.
    :return: The dataset.
    """
    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        dataset = [json.loads(line) for line in dataset_file if line.strip()]

    return dataset


def clean_dataset(dataset: list[dict[str, str | int | dict[str, list[str]]]])\
        -> list[dict[str, str | int | dict[str, list[str]]]]:
    """
    Clean the dataset.

    :param dataset: The dataset to clean.
    :return: The cleaned dataset.
    """
    cleaned_dataset = []

    dataset_as_dict = {entry["title"]: entry["definitions"] for entry in dataset}

    # Do not include entries without examples
    # If the definition starts with "то же, что <word>", try to find the definition of <word>
    # If the definition of <word> is found, use it as the definition of the current word
    # If the definition of <word> is not found, remove the entry

    for entry in dataset:
        if not entry["definitions"]:
            continue

        new_entry = entry.copy()
        new_definitions = {}

        for definition in entry["definitions"]:
            if definition.startswith("то же, что "):
                word = definition.split("то же, что ")[1]
                other_definitions = dataset_as_dict.get(word, None)
                if other_definitions and len(other_definitions) == 1:
                    the_other_definition = list(other_definitions.keys())[0]
                    print(f"Replacing {definition} with {the_other_definition}")
                    if entry["definitions"][definition]["examples"]:
                        new_definitions[the_other_definition] = entry["definitions"][definition]
            else:
                print(f"Keeping {definition}")
                print(f"Examples: {entry['definitions'][definition]['examples']}")
                if entry["definitions"][definition]["examples"]:
                    print(f"AAAA")
                    new_definitions[definition] = entry["definitions"][definition]

        new_entry["definitions"] = new_definitions

        if new_entry["definitions"]:
            cleaned_dataset.append(new_entry)

    return cleaned_dataset


def dump_dataset(dataset: list[dict[str, str | int | dict[str, list[str]]]],
                 output_path: str | Path) -> None:
    """
    Dump the dataset to the given path.

    :param dataset: The dataset to dump.
    :param output_path: The path to dump the dataset to.
    """
    with open(output_path, "w", encoding="utf-8") as output_file:
        for entry in dataset:
            output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    """
    The main function.
    """
    parser = argparse.ArgumentParser(description="Clean the dataset.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="The path to the dataset.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="The path to the output file.")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)
    cleaned_dataset = clean_dataset(dataset)

    dump_dataset(cleaned_dataset, args.output_path)
