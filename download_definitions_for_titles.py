import json
import time

from utils import print_progress_bar
from wiktionary_parser.parser import WiktionaryParser


def fetch_definitions_and_examples(titles, titles_between_saves=1000):
    parser = WiktionaryParser()
    definitions_data = {}

    successful = 0
    no_example = 0
    error = 0

    start_time = time.time()
    print_progress_bar(0, len(titles), start_time)

    for index, title in enumerate(titles):
        if index % titles_between_saves == 0 and index:
            save_definitions_to_json(definitions_data, f"tmp/definitions{index}-{successful}-{time.time()}")

        try:
            parsed_response = parser.make_request(title)
            title_data = {"definitions": []}

            found_definition_with_example = False

            for definition in parsed_response.get("definitions", []):
                if not definition.get("value", ""):
                    continue

                if definition.get("example", "") and "Отсутствует пример употребления" not in definition["example"]:
                    found_definition_with_example = True
                    title_data["definitions"].append(
                        {"value": definition["value"], "example": definition["example"]}
                    )

            if not found_definition_with_example:
                no_example += 1
                continue

            successful += 1
            definitions_data[title] = title_data
        except:
            error += 1
        print_progress_bar(index + 1, len(titles), start_time)

    print(f"\nFinished fetching... Found {successful} titles with examples of definitions, "
          f"{no_example} without, got {error} errors")
    return definitions_data


def save_definitions_to_json(definitions_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(definitions_data, file, ensure_ascii=False, indent=4)


def load_titles_from_txt(txt_file):
    try:
        with open(txt_file, 'r', encoding='utf-8') as file:
            titles = [line.strip() for line in file]
        return titles
    except FileNotFoundError:
        print(f"Error: File '{txt_file}' not found.")
        return []


if __name__ == "__main__":
    titles = load_titles_from_txt("tmp/unique_titles.txt")
    definitions_data = fetch_definitions_and_examples(titles)

    output_file = "tmp/definitions.json"
    save_definitions_to_json(definitions_data, output_file)

    print(f"Definitions and examples saved to '{output_file}'.")
