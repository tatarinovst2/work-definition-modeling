"""A module for processing templates in wiki text."""
import json
from pathlib import Path
from typing import TypedDict


class CustomMapping(TypedDict):
    """A typed dictionary for representing a custom mapping for a template."""

    title_index: int
    title: str
    description_indexes: list[int]
    starting_text: str
    ending_text: str
    arguments_count: int


class Template:
    """A class for representing a template in wiki text."""

    def __init__(self, level: int, raw_text: str, start_index: int, end_index: int):
        self.level = level
        self.raw_text = raw_text
        self.start_index = start_index
        self.end_index = end_index
        self.arguments: list[str] = []

        self._find_arguments()

    def _find_arguments(self) -> None:
        """Find the arguments of the template and stores them in self.arguments."""
        self.arguments = []
        argument = ""
        index = 0
        inside_template = False

        while index < len(self.raw_text):
            if self.raw_text[index] == "{":
                argument += self.raw_text[index]
                inside_template = True
            elif self.raw_text[index] == "}":
                argument += self.raw_text[index]
                inside_template = False
            elif self.raw_text[index] == "|" and not inside_template:
                self.arguments.append(argument)
                argument = ""
            else:
                argument += self.raw_text[index]
            index += 1
        self.arguments.append(argument)

    def get_text(self, mappings: list[CustomMapping] | None = None) -> str:
        """
        Get the text of the template.

        :param mappings: A dictionary of custom mappings to use
        :return: The text of the template
        """
        if mappings is None:
            mappings = []

        for custom_mapping in mappings:
            title_index = custom_mapping["title_index"]

            if title_index >= len(self.arguments):
                continue

            if (custom_mapping["arguments_count"] != -1 and
                    len(self.arguments) != custom_mapping["arguments_count"]):
                continue

            description_index_out_of_range = False
            for description_index in custom_mapping["description_indexes"]:
                if description_index >= len(self.arguments):
                    description_index_out_of_range = True
                    break
            if description_index_out_of_range:
                continue

            if custom_mapping["title"] == self.arguments[title_index]:
                return (custom_mapping["starting_text"] +
                        self.get_text_from_argument_indexes(
                            custom_mapping["description_indexes"], ", ") +
                        custom_mapping["ending_text"])

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
            title_index = custom_mapping.get("title_index", 0)
            if title_index >= len(self.arguments):
                continue
            if custom_mapping["title"] == self.arguments[title_index]:
                return self.arguments[custom_mapping.get("title_index", 0)]

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
                f"start_index={self.start_index}, end_index={self.end_index}, "
                f"arguments={self.arguments}")

    def __repr__(self) -> str:
        return self.__str__()


def get_templates(text: str, level: int = 0) -> list[Template]:
    """
    Get the templates in the text.

    :param text: The text to get the templates from
    :param level: The level of the current text (e.g. 1 inside one template, 2 if inside two, etc.)
    :return: A list of Template objects
    """
    templates = []
    skip_next = False

    current_templates_stack = []

    for index in range(len(text) - 1):
        if skip_next:
            skip_next = False
            continue

        if text[index] == "{" and text[index + 1] == "{":
            level += 1
            start_index = index
            current_templates_stack.append({"level": level, "start_index": start_index})
            skip_next = True
        elif text[index] == "}" and text[index + 1] == "}":
            level -= 1
            end_index = index + 2
            current_template = current_templates_stack.pop()
            template = Template(current_template["level"],
                                text[current_template["start_index"]+2:end_index-2],
                                current_template["start_index"], end_index)
            templates.append(template)
            skip_next = True

    templates = sorted(templates, key=lambda template: template.level, reverse=False)
    return templates


def replace_templates_in_text(text: str, mappings: list[CustomMapping] | None = None,
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


def load_config(filepath: str | Path) -> dict[str, list[CustomMapping] | list[str]]:
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

    return {
        "mappings": mappings,
        "templates_to_remove": mappings_dict.get("templates_to_remove", [])
    }
