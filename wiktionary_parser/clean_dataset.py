"""A module for cleaning the dataset."""
import argparse
import json
import re
from pathlib import Path

from pydantic.dataclasses import dataclass


@dataclass
class DatasetEntry:
    """A class representing a dataset entry."""

    id: int
    title: str
    definitions: dict[str, dict[str, list[str]]]


def load_dataset(dataset_path: str | Path) -> list[dict[str, str | int | dict[str, list[str]]]]:
    """
    Load the dataset of jsonl format from the given path.

    :param dataset_path: The path to the dataset.
    :return: The dataset.
    """
    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        dataset = [json.loads(line) for line in dataset_file if line.strip()]

    return dataset


def remove_non_russian(definition: str) -> str:
    """
    Remove latin characters (usually biology species).

    :param definition: The definition in the string form.
    :return: The cleaned definition.
    """
    pattern = r'\([a-zA-Z\s]+\)'

    clean_def = re.sub(pattern, '', definition)
    clean_def = re.sub(r'\s+', ' ', clean_def).strip()

    return clean_def


def clean_dataset(dataset: list[dict]) -> list[dict]:
    """
    Clean the dataset.

    :param dataset: The dataset to clean.
    :return: The cleaned dataset.
    """
    cleaned_dataset = []

    dataset_as_dict = {entry["title"]: entry["definitions"] for entry in dataset}

    for entry_dict in dataset:
        entry = DatasetEntry(id=entry_dict["id"],
                             title=entry_dict["title"],
                             definitions=entry_dict["definitions"])
        if not entry.definitions:
            continue

        new_definitions = {}
        new_entry = {"id": entry.id, "title": entry.title, "definitions": []}

        for definition in entry.definitions:
            if len(definition) > 200 or (definition[0] == "(" and definition[-1] == ")"):
                continue

            bad_markers = ["?", "=", "ru", "действие по значению",
                           "связанный, соотносящийся по значению", "свойство или состояние",
                           "страд.", "превосходная степень", "причастие от слова",
                           "сравнительная степень"]

            ignore_definition = False

            for bad_marker in bad_markers:
                if bad_marker in definition:
                    ignore_definition = True
                    break

            if ignore_definition:
                continue

            if definition.startswith("то же, что "):
                word = definition.split("то же, что ")[1]
                other_definitions = dataset_as_dict.get(word, None)
                if other_definitions and len(other_definitions) == 1:
                    the_other_definition = list(other_definitions.keys())[0]
                    if entry.definitions[definition]["examples"]:
                        new_definitions[the_other_definition] = entry.definitions[definition]
            else:
                if entry.definitions[definition]["examples"]:
                    new_definitions[remove_non_russian(definition)] = entry.definitions[definition]

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
    """Clean the dataset."""
    parser = argparse.ArgumentParser(description="Clean the dataset.")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="The path to the dataset.")
    parser.add_argument("--output-path", type=str, required=True,
                        help="The path to the output file.")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)
    cleaned_dataset = clean_dataset(dataset)

    dump_dataset(cleaned_dataset, args.output_path)


if __name__ == "__main__":
    main()
