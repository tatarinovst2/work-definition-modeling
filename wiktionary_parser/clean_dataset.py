"""A module for cleaning the dataset."""
import argparse
import json
import re
from pathlib import Path

from pydantic.dataclasses import dataclass

from utils import parse_path, WIKTIONARY_PARSER_DIR


@dataclass
class DatasetEntry:
    """A class representing a dataset entry."""

    id: int
    title: str
    definitions: dict[str, dict[str, list[str]]]


@dataclass
class LimitingMarker:
    """A class for representing the config that limits the number of entries with a marker."""

    marker: str
    leave_each: int


@dataclass
class WiktionaryCleaningConfig:
    """A class for representing the cleaning config."""

    remove_latin_in_parenthesis: bool
    remove_missed_tags: bool
    max_definition_character_length: int
    remove_entries_without_examples: bool
    throw_out_definition_markers: list[str]
    markers_for_limiting: list[LimitingMarker]
    tags_to_remove: list[str]


def load_dataset(dataset_path: str | Path) -> list[dict[str, str | int | dict[str, list[str]]]]:
    """
    Load the dataset of jsonl format from the given path.

    :param dataset_path: The path to the dataset.
    :return: The dataset.
    """
    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        dataset = [json.loads(line) for line in dataset_file if line.strip()]

    return dataset


def load_config(config_path: str | Path) -> WiktionaryCleaningConfig:
    """
    Load the cleaning config from the given path.

    :param config_path: The path to the config.
    :return: The cleaning config.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    return WiktionaryCleaningConfig(**config)


def process_definition(definition: str, title: str, config: WiktionaryCleaningConfig) -> str:
    """
    Process the definition by removing unnecessary parts and formatting it.

    :param definition: The definition in the string form.
    :param title: The title of the entry.
    :param config: The cleaning config.
    :return: The cleaned definition.
    """
    if config.remove_latin_in_parenthesis:
        definition = re.sub(r'\([a-zA-Z\s]+\)', '', definition)

    if config.remove_missed_tags:
        for tag in config.tags_to_remove:
            if re.match(fr"\b{re.escape(tag)},?", definition):
                definition = re.sub(fr"\b{re.escape(tag)},?", "", definition)

    definition = re.sub(r'^\s?(?:[и,;]|или)[\s,.]', '', definition)
    definition = re.sub(r'\s+', ' ', definition).strip()

    if "то же, что" in definition.lower():
        right_part = definition.lower().split("то же, что")[1].strip()
        if right_part[:2].lower() == title[:2].lower() or not right_part:
            return ""
        definition = right_part[0].upper() + right_part[1:]

    start_tags_to_be_removed = ["женск. к", "женское к"]

    for tag in start_tags_to_be_removed:
        if tag in definition.lower():
            if not "; " in definition:
                return ""
            definition = "; ".join(definition.split("; ")[1:])
            if tag in definition or len(definition.split()) < 2:
                return ""

    if "химический элемент с атомным номером" in definition.lower():
        definition = re.sub(r" с атомным номером \d+", "", definition)

    if re.search(rf"\b{re.escape(title.lower())}\b", definition.lower()) or not definition:
        return ""

    return definition


def should_ignore_definition(definition: str, config: WiktionaryCleaningConfig,
                             marker_counts: dict) -> bool:
    """
    Check if the definition should be ignored.

    :param definition: The definition to check.
    :param config: The cleaning config.
    :param marker_counts: The counts of markers.
    :return: True if the definition should be ignored, otherwise False.
    """
    if len(definition) > config.max_definition_character_length or len(definition) < 3:
        return True
    if definition.startswith("(") and definition.endswith(")"):
        return True

    for bad_marker in config.throw_out_definition_markers:
        if bad_marker in definition:
            return True

    for marker in marker_counts:
        if marker in definition:
            marker_counts[marker]["count"] += 1
            if marker_counts[marker]["count"] >= marker_counts[marker]["leave_each"]:
                marker_counts[marker]["count"] = 0
                return False
            return True

    return False


def clean_dataset(dataset: list[dict], config: WiktionaryCleaningConfig) -> list[dict]:
    """
    Clean the dataset.

    :param dataset: The dataset to clean.
    :param config: The cleaning config.
    :return: The cleaned dataset.
    """
    cleaned_dataset = []

    markers_for_limiting = config.markers_for_limiting
    marker_counts = {marker.marker: {"count": 0, "leave_each": marker.leave_each}
                     for marker in markers_for_limiting}

    for entry_dict in dataset:
        entry = DatasetEntry(id=entry_dict["id"],
                             title=entry_dict["title"],
                             definitions=entry_dict["definitions"])
        if not entry.definitions:
            continue

        # We are interested not in suffixes and prefixes, but in words
        if entry.title.startswith("-") or entry.title.endswith("-"):
            continue

        new_definitions = {}
        new_entry = {"id": entry.id, "title": entry.title, "definitions": []}

        for definition in entry.definitions:
            if should_ignore_definition(definition, config, marker_counts):
                continue

            if (config.remove_entries_without_examples and
                    not entry.definitions[definition]["examples"]):
                continue

            examples = entry.definitions[definition]

            processed_definition = process_definition(definition, entry.title, config)

            if " " not in processed_definition:
                continue

            if processed_definition:
                new_definitions[processed_definition] = examples

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

    config = load_config(WIKTIONARY_PARSER_DIR / "wiktionary_cleaning_config.json")
    dataset = load_dataset(parse_path(args.dataset_path))
    cleaned_dataset = clean_dataset(dataset, config)

    dump_dataset(cleaned_dataset, parse_path(args.output_path))


if __name__ == "__main__":
    main()
