"""A module for cleaning the dataset."""
import argparse
import re

from parse_mas_utils import (dump_dataset, load_cleaning_config, load_dataset, MasCleaningConfig,
                             parse_path, ROOT_DIR)
from pydantic.dataclasses import dataclass


@dataclass
class DatasetEntry:
    """A class representing a dataset entry."""

    id: int
    title: str
    definitions: dict[str, dict[str, list[str]]]


def should_ignore_definition(definition: str, config: MasCleaningConfig) -> bool:
    """
    Check if the definition should be ignored.

    :param definition: The definition to check.
    :param config: The cleaning config.
    :return: True if the definition should be ignored, otherwise False.
    """
    if len(definition) > config.max_definition_character_length:
        return True

    for bad_marker in config.throw_out_definition_markers:
        if bad_marker in definition.lower():
            return True

    return False


def process_definition(definition: str, config: MasCleaningConfig) -> str:
    """
    Process the definition.

    :param definition: The definition to process.
    :param config: The cleaning config.
    :return: The processed definition.
    """
    for key, value in config.replace.items():
        definition = re.sub(fr"\b{re.escape(key)}", value, definition)

    if definition[0].isupper():
        definition = definition[0].lower() + definition[1:]

    if definition.endswith("."):
        definition = definition[:-1]

    return definition


def clean_dataset(dataset: list[dict], config: MasCleaningConfig) -> list[dict]:
    """
    Clean the dataset.

    :param dataset: The dataset to clean.
    :param config: The cleaning config.
    :return: The cleaned dataset.
    """
    cleaned_dataset = []

    for entry_dict in dataset:
        entry = DatasetEntry(id=entry_dict["id"],
                             title=entry_dict["title"],
                             definitions=entry_dict["definitions"])

        new_definitions = {}
        new_entry = {"id": entry.id, "title": entry.title, "definitions": []}

        for definition in entry.definitions:
            if should_ignore_definition(definition, config):
                continue

            if (config.remove_entries_without_examples and
                    not entry.definitions[definition]["examples"]):
                continue

            examples = entry.definitions[definition]

            if "то же, что" in definition.lower():
                right_part = definition.lower().split("то же, что")[1].strip()
                if right_part[:2].lower() == entry.title[:2].lower() or not right_part:
                    continue
                definition = right_part[0].upper() + right_part[1:]

            # tags_to_split = ["женск. к", "уменьш. к", "ласк. к", "уменьш.-ласк. к",
            #                  "состояние по глаг."]
            #
            # for tag_to_split in tags_to_split:
            #     if tag_to_split in definition.lower():
            #         right_part = definition.lower().split(tag_to_split)[-1]
            #         if len(right_part.split()) < 4 or ";" not in definition:
            #             continue
            #         print(f"Updating definition: {definition}")
            #         definition = definition.split(";")[-1]
            #         print(f"to {definition}")

            # if "женск. к" in definition.lower():
            #     if ";" not in definition:
            #         continue
            #     if len(right_part.split()) < 4 or ";" not in right_part:
            #         print(f"Skipped: {definition}")
            #         continue
            #     definition = definition.split(";")[-1]
            # elif "уменьш. к" in definition.lower():
            #     right_part = definition.lower().split("уменьш. к")[-1]
            #     if len(right_part.split()) < 4 or ";" not in right_part:
            #         continue
            #     definition = definition.split(";")[-1]
            # elif "ласк. к" in definition.lower():
            #     right_part = definition.lower().split("ласк. к")[1]
            #     if 4 < len(right_part.split()) <= 5:
            #         print(definition)
            #     if len(right_part.split()) <= 5 or ";" not in right_part:
            #         continue
            #     definition = definition.split(";")[-1]
            # elif "уменьш.-ласк. к" in definition.lower():
            #     right_part = definition.lower().split("уменьш.-ласк. к")[1]
            #     if 4 < len(right_part.split()) <= 5:
            #         print(definition)
            #     if len(right_part.split()) <= 5 or ";" not in right_part:
            #         continue
            #     definition = definition.split(";")[-1]
            # elif "состояние по глаг." in definition.lower():
            #     right_part = definition.lower().split("состояние по глаг.")[1]
            #     if 4 < len(right_part.split()) <= 5:
            #         print(definition)
            #     if len(right_part.split()) <= 5 or ";" not in right_part:
            #         continue
            #     definition = definition.split(";")[-1]

            if re.search(r":,+\.?", definition) or re.search(r":\.", definition):
                continue

            if not definition:
                continue

            definition = process_definition(definition, config)

            if " " not in definition:
                continue

            if re.search(rf"\b{re.escape(entry.title.lower())}\b", definition.lower()):
                continue

            if not definition:
                continue

            new_definitions[definition] = examples

        new_entry["definitions"] = new_definitions

        if new_entry["definitions"]:
            cleaned_dataset.append(new_entry)

    return cleaned_dataset


def main() -> None:
    """Clean the dataset."""
    parser = argparse.ArgumentParser(description="Clean the dataset.")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="The path to the dataset.")
    parser.add_argument("--output-path", type=str, required=True,
                        help="The path to the output file.")

    args = parser.parse_args()

    config = load_cleaning_config(ROOT_DIR / "mas_parser" / "mas_cleaning_config.json")
    dataset = load_dataset(parse_path(args.dataset_path))
    cleaned_dataset = clean_dataset(dataset, config)

    dump_dataset(cleaned_dataset, parse_path(args.output_path))


if __name__ == "__main__":
    main()
