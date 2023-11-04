"""A script that parses the Wiktionary xml.bz2 dump file and extracts articles."""
import bz2
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import wikitextparser as wtp

from wiktionary_parser.template_parsing import (load_config, pop_templates_in_text,
                                                replace_templates_in_text)
from wiktionary_parser.utils import clean_text

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


def parse_dump(input_filepath: str | Path, output_filepath: str | Path,
               parser_config: dict) -> None:
    """
    Parse the Wiktionary xml.bz2 dump file and extract articles.

    :param input_filepath: The path to the Wikipedia dump file.
    :param output_filepath: The path to the output jsonl file.
    :param parser_config: The parser config.
    """
    with bz2.open(input_filepath, 'rb') as xml_file:
        for index, (_, element) in enumerate(ET.iterparse(xml_file)):
            process_element(element, output_filepath, parser_config)
            print(f"\rProcessed {index} XML elements...", end="")


def process_element(element: ET.Element, output_filepath: str | Path, parser_config: dict) -> None:
    """
    Check if the XML element is an article and extract its id, title and definitions.

    :param element: The XML element to process.
    :param output_filepath: The path to the output jsonl file.
    :param parser_config: The parser config.
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

    definitions = parse_wiki(wiki, parser_config)
    if not definitions:
        element.clear()
        return

    with open(output_filepath, "a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(
            {"id": identifier, "title": title, "definitions": definitions},
            ensure_ascii=False) + "\n")

    element.clear()


def get_sections(wiki_text: wtp.WikiText, section_title: str) -> list[wtp.Section]:
    """
    Get a section from the wiki text by its title.

    :param wiki_text: the wiki text to get the section from.
    :param section_title: the title of the section to get.
    :return: The section with the given title or None if it doesn't exist.
    """
    result_sections = []

    sections = wiki_text.get_sections()
    for section in sections:
        if section_title in str(section.title).strip():
            result_sections.append(section)

    return result_sections


def parse_wiki(wiki: str, parser_config: dict) -> dict[str, dict[str, list[str]]] | None:
    """
    Parse the wiki text of an article and extract definitions.

    :param wiki: The wiki text of an article.
    :param parser_config: The parser config.
    :return: The definitions of the article or None if the article doesn't have any definitions.
    """
    ru_sections = [section for section in wtp.parse(wiki).get_sections()
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

            definition, example_templates = pop_templates_in_text(
                definition_wiki_text.plain_text(
                replace_templates=False).replace("\n", ""),
                "пример",
                mappings=parser_config["mappings"])
            definition = replace_templates_in_text(
                definition,
                mappings=parser_config["mappings"],
                templates_to_remove=parser_config["templates_to_remove"])
            definition = clean_text(definition, words_to_remove=["также", "и"])
            if not definition:
                continue

            examples = []

            for example_template in example_templates:
                example_text = replace_templates_in_text(
                    example_template.get_text(mappings=parser_config["mappings"]),
                    mappings=parser_config["mappings"],
                    templates_to_remove=parser_config["templates_to_remove"])
                example_text = clean_text(example_text, words_to_remove=["также", "и"])
                if not example_text or example_text == "пример":
                    continue
                examples.append(example_text)

            if not examples:
                continue

            definitions[definition] = {"examples": examples}

    return definitions


def validate_filepaths(input_filepath: str | Path, output_filepath: str | Path) -> None:
    """
    Create the directories and check if the files exist.

    :param input_filepath: The path to the Wikipedia dump file.
    :param output_filepath: The path to the output jsonl file.
    :raises FileNotFoundError: If the input file doesn't exist.
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
    output_definitions_filepath = WIKTIONARY_PARSER_DIR / "data" / "definitions.jsonl"
    config_filepath = WIKTIONARY_PARSER_DIR / "wiktionary_parser_config.json"

    config = load_config(config_filepath)

    validate_filepaths(dump_filepath, output_definitions_filepath)

    parse_dump(dump_filepath, output_definitions_filepath, config)
