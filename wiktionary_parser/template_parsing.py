"""A module for processing templates in wiki text."""
import json
from dataclasses import dataclass
from pathlib import Path

import wikitextparser as wtp


@dataclass
class CustomMapping:
    """A data class for representing a custom mapping for a template."""

    title_index: int
    title: str
    description_indexes: list[int]
    starting_text: str
    ending_text: str
    arguments_count: int


@dataclass
class ParserConfig:
    """A data class for representing the parser config."""

    mappings: list[CustomMapping]
    templates_to_remove: list[str]


def load_config(filepath: str | Path) -> ParserConfig:
    """
    Load the config from the given filepath.

    :param filepath: The filepath to load the config from
    :return: The config
    """
    mappings = []

    with open(filepath, "r", encoding="utf-8") as json_file:
        mappings_dict = json.load(json_file)

    for custom_mapping in mappings_dict.get("mappings", []):
        mapping = CustomMapping(title_index=custom_mapping["title_index"],
                                title=custom_mapping["title"],
                                description_indexes=custom_mapping["description_indexes"],
                                starting_text=custom_mapping.get("starting_text", ""),
                                ending_text=custom_mapping.get("ending_text", ""),
                                arguments_count=custom_mapping.get("arguments_count", -1))

        mappings.append(mapping)

    templates_to_remove = mappings_dict.get("templates_to_remove", [])

    return ParserConfig(mappings=mappings, templates_to_remove=templates_to_remove)


class Template:
    """A class for representing a template in wiki text."""

    def __init__(self, level: int, raw_text: str, arguments: list[str]):
        self.level = level
        self.raw_text = raw_text
        self.arguments: list[str] = arguments

    def get_text(self, mappings: list[CustomMapping] | None = None) -> str:
        """
        Get the text of the template.

        :param mappings: A dictionary of custom mappings to use
        :return: The text of the template
        """
        if mappings is None:
            mappings = []

        for custom_mapping in mappings:
            if custom_mapping.title_index >= len(self.arguments):
                continue

            if (custom_mapping.arguments_count != -1 and
                    len(self.arguments) != custom_mapping.arguments_count):
                continue

            description_index_out_of_range = False
            for description_index in custom_mapping.description_indexes:
                if description_index >= len(self.arguments):
                    description_index_out_of_range = True
                    break
            if description_index_out_of_range:
                continue

            if custom_mapping.title == self.arguments[custom_mapping.title_index]:
                return (custom_mapping.starting_text +
                        self.get_text_from_argument_indexes(
                            custom_mapping.description_indexes, ", ") +
                        custom_mapping.ending_text)

        if len(self.arguments) == 1:
            return self.arguments[0]

        return self.get_text_from_argument_indexes([1], ", ")

    def get_title(self, mappings: list[CustomMapping] | None = None) -> str:
        """
        Get the title of the template.

        :param mappings: A dictionary of custom mappings to use
        :return: The title of the template
        """
        if mappings is None:
            mappings = []

        for custom_mapping in mappings:
            title_index = custom_mapping.title_index
            if title_index >= len(self.arguments):
                continue
            if custom_mapping.title == self.arguments[title_index]:
                return self.arguments[custom_mapping.title_index]

        return self.arguments[0]

    def get_text_from_argument_indexes(self, indexes: list[int], separator: str) -> str:
        """
        Get the text of the template from the given argument indexes.

        :param indexes: The indexes of the arguments to get the text from
        :param separator: The separator to use between the arguments
        :return: The text of the template from the given argument indexes
        """
        return separator.join([self.arguments[index] for index in indexes])

    def __str__(self) -> str:
        return (f"Template(level={self.level}, raw_text={self.raw_text}, "
                f"arguments={self.arguments}")

    def __repr__(self) -> str:
        return self.__str__()


def get_templates(text: str, level: int = 0) -> list[Template]:
    """
    Get the templates in the text.

    :param text: The text to get the templates from
    :param level: The level of the current text
    :return: A list of Template objects
    """
    templates = []
    wiki_text = wtp.WikiText(text)

    for wiki_template in wiki_text.templates:
        arguments = ([wiki_template.name] +
                     [argument.string[1:] for argument in wiki_template.arguments if
                      argument.string != "|lang=ru"]) # lang=ru can be anywhere in the template
        # so we remove it for consistency

        template = Template(level + wiki_template.nesting_level, wiki_template.string[2:-2],
                            arguments)
        templates.append(template)

    templates = sorted(templates, key=lambda template: template.level, reverse=False)
    return templates


def replace_templates_with_text(text: str, mappings: list[CustomMapping] | None = None,
                                templates_to_remove: list[str] | None = None) -> str:
    """
    Replace the templates in the text with their values.

    :param text: The text to replace the templates in
    :param mappings: A list of custom mappings to use
    :param templates_to_remove: A list of templates to be fully removed
    :return: The text with the templates replaced
    """
    if mappings is None:
        mappings = []

    if templates_to_remove is None:
        templates_to_remove = []

    templates = get_templates(text)
    for template in templates:
        if template.get_title(mappings) in templates_to_remove:
            text = text.replace("{{" + template.raw_text + "}}", "")
            continue

        text = text.replace("{{" + template.raw_text + "}}", template.get_text(mappings))

    return text


def pop_templates_in_text(text: str, title_to_pop: str,
                          mappings: list[CustomMapping] | None = None)\
        -> tuple[str, list[Template]]:
    """
    Pops the templates in the text with the given title.

    :param text: The text to pop the templates in
    :param title_to_pop: The title of the templates to pop
    :param mappings: A list of custom mappings to use
    :return: The text with the templates popped and a list of the popped templates
    """
    if mappings is None:
        mappings = []

    templates = get_templates(text)
    popped_templates = []
    for template in templates:
        if template.get_title(mappings) == title_to_pop:
            text = text.replace("{{" + template.raw_text + "}}", "")
            popped_templates.append(template)

    return text, popped_templates
