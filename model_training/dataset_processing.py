"""A module for functions that process data for the model."""
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict


def prepare_row(row: dict) -> dict:
    """
    Converts a row from the dataset to a readable format.
    :param row: dict - the row to prepare.
    :return: dict - the prepared row.
    """
    input_text = f"<LM>{row['context']}\n Определение слова \"{row['word']}\": "
    target_text = row['definition']
    return {"input_text": input_text, "target_text": target_text}


def prepare_dataset(filepath: str | Path) -> DatasetDict:
    """
    Loads and creates a dataset from a jsonl file.
    :param filepath: str | Path - the path to the jsonl file.
    :return: DatasetDict - the dataset with train and test splits.
    """
    rows = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            json_line = json.loads(line)
            for definition in json_line["definitions"]:
                examples = json_line["definitions"][definition]
                for example in examples:
                    rows.append({"word": json_line["title"],
                                 "definition": definition,
                                 "context": example})

    df = pd.DataFrame(rows)

    df = df[df["word"].str.len() >= 3]

    prepared_rows = [prepare_row(row.to_dict()) for _, row in df.iterrows()]
    new_df = pd.DataFrame(prepared_rows)

    created_dataset = Dataset.from_pandas(new_df)
    created_dataset = created_dataset.train_test_split(test_size=0.2, seed=42)

    return created_dataset
