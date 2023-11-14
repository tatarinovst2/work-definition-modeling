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


def prepare_dataset(filepath: str | Path, test_dataset_output_path: str | Path = "",
                    random_state: int = 42, debug_mode: bool = False) -> DatasetDict:
    """
    Load and create a dataset from a jsonl file.

    :param filepath: The path to the jsonl file.
    :param test_dataset_output_path: The path to the output file for the test dataset.
    :param random_state: The random state for the train/test/validation split.
    :param debug_mode: Whether to run in debug mode, i.e. use only a small subset of the dataset.
    :return: The dataset with train, test and validation splits.
    """
    rows = load_dataset(filepath)

    df = pd.DataFrame(rows)
    df = df[df["word"].str.len() >= 3]

    if debug_mode:
        df = df.sample(100, random_state=random_state)

    new_df = pd.DataFrame([prepare_row(row.to_dict()) for _, row in df.iterrows()])

    train_df, test_df = train_test_split(new_df, test_size=0.1, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=0.9*0.1,
                                        random_state=random_state)

    created_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df),
        "validation": Dataset.from_pandas(val_df)
    })

    if test_dataset_output_path:
        save_test_dataset(created_dataset["test"], test_dataset_output_path)

    return created_dataset


def load_dataset(filepath: str | Path) -> list[dict]:
    """
    Load a dataset from a jsonl file.

    :param filepath: The path to the jsonl file.
    :return: The dataset.
    """
    rows = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            json_line = json.loads(line)
            for definition in json_line["definitions"]:
                examples = json_line["definitions"][definition]["examples"]
                for example in examples:
                    rows.append({"word": json_line["title"],
                                 "definition": definition,
                                 "context": example})

    return rows


def save_test_dataset(dataset: Dataset, output_file: str | Path) -> None:
    """
    Save the test dataset to a JSON Lines file.

    :param dataset: The dataset to save.
    :param output_file: The path to the output file.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        for row in dataset:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
