"""A module for functions that process data for the model."""
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def prepare_row(row: dict) -> dict:
    """
    Convert a row from the dataset to a readable format.

    :param row: The row to prepare.
    :return: The prepared row.
    """
    input_text = f"<LM>Контекст: \"{row['context']}\" Определение слова \"{row['word']}\": "
    target_text = row['definition']
    return {"input_text": input_text, "target_text": target_text}


def prepare_dataset(filepath: str | Path) -> DatasetDict:
    """
    Load and create a dataset from a jsonl file.

    :param filepath: The path to the jsonl file.
    :return: The dataset with train, test and validation splits.
    """
    rows = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            json_line = json.loads(line)
            for definition in json_line["definitions"]:
                examples = json_line["definitions"][definition]["examples"]
                for example in examples:
                    rows.append({"word": json_line["title"],
                                 "definition": definition,
                                 "context": example})

    df = pd.DataFrame(rows)

    df = df[df["word"].str.len() >= 3]

    prepared_rows = [prepare_row(row.to_dict()) for _, row in df.iterrows()]
    new_df = pd.DataFrame(prepared_rows)

    train_df, test_df = train_test_split(new_df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.9*0.1, random_state=42)

    created_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df),
        "validation": Dataset.from_pandas(val_df)
    })

    save_test_dataset(created_dataset["test"], "test.jsonl")

    return created_dataset


def save_test_dataset(dataset: Dataset, output_file: str | Path) -> None:
    """
    Save the test dataset to a JSON Lines file.

    :param dataset: The dataset to save.
    :param output_file: The path to the output file.
    """
    with open(output_file, "w", encoding="utf-8") as file:
        for row in dataset:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
