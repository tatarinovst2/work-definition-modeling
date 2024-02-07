"""Convert raw annotations to a dataset for inference."""
import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rushifteval_utils import AnnotatedWord, parse_path


def load_tsv_dataset(file_path: str | Path) -> list[AnnotatedWord]:
    """
    Load a TSV dataset and parses each row into an AnnotatedWord dataclass.

    :param file_path: The path to the TSV file.
    :return: A list of AnnotatedWord instances.
    """
    dataset = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            annotated_word = AnnotatedWord(
                word=row['word'],
                sent1=row['sent1'],
                sent2=row['sent2'],
                mean=float(row['mean'])
            )
            dataset.append(annotated_word)
    return dataset


def create_input_text(sentence: str, word: str) -> str:
    """
    Create the input text for the model inference.

    :param sentence: The sentence in which the word is used.
    :param word: The target word.
    :return: Formatted input text for the model.
    """
    sentence = sentence.replace("<b><i>", "").replace("</i></b>", "")
    return f"<LM> Контекст: \"{sentence}\" Определение слова \"{word}\": "


def prepare_dataset_for_inference(dataset: list[AnnotatedWord]) -> list[dict[str, Any]]:
    """
    Prepare the dataset for inference by creating an 'input_text' field and splitting examples.

    :param dataset: The original dataset loaded from the TSV file.
    :return: A list of dictionaries with the prepared data for inference.
    """
    prepared_data = []
    for idx, annotated_word in enumerate(dataset):
        for sentence_id, sentence in enumerate([annotated_word.sent1, annotated_word.sent2], 1):
            prepared_data.append({
                "id": idx,
                "sentence_id": sentence_id,
                "word": annotated_word.word,
                "sentence": sentence,
                "input_text": create_input_text(sentence, annotated_word.word)
            })
    return prepared_data


def save_as_json(prepared_data: list[dict[str, Any]], output_path: str | Path) -> None:
    """
    Save the prepared dataset as a JSON file.

    :param prepared_data: The dataset prepared for inference.
    :param output_path: The path where the JSON file will be saved.
    """
    parsed_output_path = parse_path(output_path)

    if not parsed_output_path.parent.exists():
        parsed_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(parsed_output_path, 'w', encoding='utf-8') as output_file:
        for record in prepared_data:
            json_record = json.dumps(record, ensure_ascii=False)
            output_file.write(json_record + '\n')


def main() -> None:
    """Process raw annotations."""
    parser = argparse.ArgumentParser(
        description="Prepare shifteval dataset for inference.")

    parser.add_argument("dataset_path",
                        type=str,
                        help="The path to the dataset. It can be either a .tsv file or"
                             "a directory containing multiple .tsv files.")
    parser.add_argument("output_path",
                        type=str,
                        help="The output path. Either a .tsv file or a directory.")

    args = parser.parse_args()

    dataset_path = parse_path(args.dataset_path)
    output_path = parse_path(args.output_path)

    if not dataset_path.exists():
        raise ValueError("This dataset does not exist.")

    if dataset_path.suffix != output_path.suffix:
        raise ValueError("Both dataset_path and output_path should be either .tsv files"
                         "or directories.")

    if dataset_path.is_dir():
        for path in dataset_path.glob("*.tsv"):
            dataset = load_tsv_dataset(path)
            dataset_for_inference = prepare_dataset_for_inference(dataset)
            save_as_json(dataset_for_inference, parse_path(f"{output_path}/{path.stem}.jsonl"))
    else:
        dataset = load_tsv_dataset(dataset_path)
        dataset_for_inference = prepare_dataset_for_inference(dataset)
        save_as_json(dataset_for_inference, output_path)


if __name__ == "__main__":
    main()
