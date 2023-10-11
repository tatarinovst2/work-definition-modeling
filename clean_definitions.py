from pathlib import Path
import csv
import re


def clean_definitions(csv_filepath: str | Path) -> None:
    new_csv_rows = []

    with open(csv_filepath, "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            word, context, definition = row

            if not definition.strip() or not context.strip():
                continue

            pattern = r"([А-Я]\.\s?[А-Я]\.)"
            match_index = re.search(pattern, context)
            if match_index:
                context = context[:match_index.start()].strip()

            new_csv_rows.append([word, context, definition])

    with open(csv_filepath, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["word", "context", "definition"])
        csv_writer.writerows(new_csv_rows)


if __name__ == "__main__":
    clean_definitions("tmp/definitions.csv")