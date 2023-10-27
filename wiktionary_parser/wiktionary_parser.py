"""A script that parses the Wiktionary xml.bz2 dump file and extracts articles."""
import bz2
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict

import wikitextparser as wtp

WIKTIONARY_PARSER_DIR = Path(__file__).parent
ROOT_XML_TAG = "{http://www.mediawiki.org/xml/export-0.10/}"

TAGS: dict[str, str] = {
    "text": ROOT_XML_TAG + "revision//" + ROOT_XML_TAG + "text",
    "title": ROOT_XML_TAG + "title",
    "redirect": ROOT_XML_TAG + "redirect",
    "namespace": ROOT_XML_TAG + "ns",
    "id": ROOT_XML_TAG + "id"
}

ARTICLE_NAMESPACE: str = "0"


class TemplateDict(TypedDict):
    """
    A dictionary corresponding to templates in wikitextparser
    """
    text: str
    start_index: int
    end_index: int


def parse_dump(input_filepath: str | Path, output_filepath: str | Path) -> None:
    """
    Parses the Wiktionary xml.bz2 dump file and extracts articles.
    :param input_filepath: str | Path - the path to the Wikipedia dump file.
    :param output_filepath: str | Path - the path to the output jsonl file.
    :return: None
    """
    with bz2.open(input_filepath, 'rb') as xml_file:
        for index, (_, element) in enumerate(ET.iterparse(xml_file)):
            process_element(element, output_filepath)
            print(f"\rProcessed {index} XML elements...", end="")


def process_element(element: ET.Element, output_filepath: str | Path) -> None:
    """
    Processes a single XML element. Checks if the element is an article
    and extracts its id, title and definitions.
    :param element: ET.Element - the XML element to process.
    :param output_filepath: str | Path - the path to the output jsonl file.
    :return: None
    """
    namespace_element = element.find(TAGS["namespace"])
    if namespace_element is None:
        return

    if not "page" in element.tag or namespace_element.text != ARTICLE_NAMESPACE:
        return

    id_element = element.find(TAGS["id"])
    title_element = element.find(TAGS["title"])
    wiki_element = element.find(TAGS["text"])

    if id_element is None or title_element is None or wiki_element is None:
        return

    if id_element.text is None or title_element.text is None or wiki_element.text is None:
        return

    identifier: int = int(id_element.text)
    title: str = title_element.text
    wiki: str = wiki_element.text

    definitions = parse_wiki(wiki)
    if not definitions:
        return

    with open(output_filepath, "a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(
            {"id": identifier, "title": title, "definitions": definitions},
            ensure_ascii=False) + "\n")


def get_sections(wiki_text: wtp.WikiText, section_title: str) -> list[wtp.Section]:
    """
    Gets a section from the wiki text by its title.
    :param wiki_data: wtp.WikiText - the wiki text to get the section from.
    :param section_title: str - the title of the section to get.
    :return: wtp.Section | None - the section with the given title or None if it doesn't exist.
    """
    result_sections = []

    sections = wiki_text.get_sections()
    for section in sections:
        if section_title in str(section.title).strip():
            result_sections.append(section)

    return result_sections


# replace_templates False fully leaves the template in the text. True removes it completely.
# We need to leave the value of the template, but remove the brackets and the template name.
def replace_templates_in_example(text: str) -> str:
    """
    Replaces templates in the example text with their values.
    :param text: str - the example text to replace templates in.
    :return: str - the example text with replaced templates.
    """
    return re.sub(r"{{[\w-]*?\|?([\w-]+)}}", lambda match: match.group(1), text).strip()


def replace_templates_in_definition(text: str) -> str:
    """
    Replaces templates in the definition text with their values.
    :param text: str - the definition text to replace templates in.
    :return: str - the definition text with replaced templates.
    """
    def get_template(text: str, template_start: str) -> TemplateDict | None:
        """
        Gets the template from the text.
        :param text: str - the text to get the template from.
        :param template_start: str - the template start string.
        :return: dict[str, str | int] | None - the template or None if it doesn't exist.
        """
        level = 0
        found_template = False

        start_index = 0
        end_index = 0

        skip_next = False

        for index in range(len(text) - 1):
            if skip_next:
                skip_next = False
                continue

            if text[index: index + len(template_start)] == template_start:
                level += 1
                found_template = True
                start_index = index
            elif found_template and text[index: index + 2] == "{{":
                skip_next = True
                level += 1
            elif found_template and text[index: index + 2] == "}}":
                skip_next = True
                level -= 1
                if level == 0:
                    end_index = index
                    break

        if not found_template:
            return None

        template_text = text[start_index + len(template_start): end_index]

        result_dict: TemplateDict = {
            "text": template_text,
            "start_index": start_index,
            "end_index": end_index
        }

        return result_dict

    templates_to_replace = ["{{t:=|", "{{помета|", "{{===|"]

    for template_to_replace in templates_to_replace:
        template = get_template(text, template_to_replace)
        if not template:
            continue

        template_text = template["text"]

        if template_to_replace == "{{t:=|":
            text = (text[:template["start_index"]] + "то же, что " + template_text +
                    text[template["end_index"] + 2:])
        elif template_to_replace == "{{помета|":
            end_text = text[template["end_index"] + 2:]
            if not end_text.startswith((":", ";", ",", ".")):
                end_text = ":" + end_text
            text = text[:template["start_index"]] + template_text + end_text
        else:
            text = text[:template["start_index"]] + template_text + text[template["end_index"] + 2:]

    # Leaving abbreviations as they are
    text = re.sub(r"{{([.\s\w-]+)\|ru\|([.\s\w-]+)}}",
                  lambda match: f"{match.group(1)} {match.group(2)}", text)
    text = re.sub(r"{{([.\s\w-]+)\|ru}}", lambda match: match.group(1), text)
    # Leave regionalisms as they are
    text = re.sub(r"{{рег\.\|([.\s\w-]+)}}", lambda match: f"рег. {match.group(1)}", text)
    # Leave templates with a single argument as they are
    text = re.sub(r"{{([.\s\w-]+)}}", lambda match: match.group(1), text)
    # Getting rid of the rest of the templates
    text = wtp.WikiText(text).plain_text()
    # Get rid of |
    text = text.replace("|", "; ")
    # Clamp the spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_wiki(wiki: str) -> dict[str, list[str]] | None:
    """
    Parses the wiki text of an article and extracts definitions.
    :param wiki: str - the wiki text of an article.
    :return: dict[str, list[str]] | None - the definitions of the article or None
    if it doesn't have any.
    """
    wiki_data = wtp.parse(wiki)
    sections = wiki_data.get_sections()
    ru_sections = [section for section in sections
                   if section.level == 1 and "{{-ru-}}" in section.title]
    if not ru_sections:
        return None

    ru_section = ru_sections[0]

    definitions_sections = get_sections(ru_section, "Значение")

    definitions = {}

    for definitions_section in definitions_sections:
        definitions_list = definitions_section.get_lists()
        if not definitions_list:
            continue

        for definition_item in definitions_list[0].items:
            definition_wiki_text = wtp.WikiText(definition_item)

            definition = replace_templates_in_definition(definition_wiki_text.plain_text(
                replace_templates=False))
            definition = definition.replace("\xa0", " ")
            if not definition:
                continue

            examples = []
            for template in definition_wiki_text.templates:
                if template.name == "пример":
                    if not template.arguments:
                        continue
                    example_text = template.arguments[0].plain_text(replace_templates=False)
                    example_text = replace_templates_in_example(example_text)
                    example_text = (example_text.replace("\xa0", " ").
                                    replace("|", ""))
                    if not example_text:
                        continue
                    examples.append(example_text)

            if not examples:
                continue

            definitions[definition] = examples

    return definitions


def validate_filepaths(input_filepath: str | Path, output_filepath: str | Path) -> None:
    """Creates the directories and checks if the files exist.
    :param input_filepath: str | Path - the path to the Wikipedia dump file.
    :param output_filepath: str | Path - the path to the output jsonl file.
    :return: None
    """
    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)

    if not input_filepath.exists():
        raise FileNotFoundError(f"Input file {input_filepath} does not exist.")

    if not output_filepath.parent.exists():
        output_filepath.parent.mkdir(parents=True)

    if output_filepath.exists():
        output_filepath.unlink()


if __name__ == "__main__":
    dump_filepath = WIKTIONARY_PARSER_DIR / "data" / "ruwiktionary-latest-pages-articles.xml.bz2"
    output_definitions_filepath = WIKTIONARY_PARSER_DIR / "data" / 'definitions.jsonl'

    validate_filepaths(dump_filepath, output_definitions_filepath)

    parse_dump(dump_filepath, output_definitions_filepath)
