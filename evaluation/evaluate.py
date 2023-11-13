"""A module for evaluation of the model."""
import json
from pathlib import Path

from model_training.metrics import get_bleu_score, get_rouge_score, get_bert_score


def load_target_pred_dataset(dataset_path: str | Path, target_field: str,
                             pred_field: str) -> tuple[list[str], list[str]]:
    """
    Load the target and predicted texts from a dataset.

    :param dataset_path: The path to the JSON Lines dataset.
    :param target_field: The name of the field containing the target texts.
    :param pred_field: The name of the field containing the predicted texts.
    :return: The target texts and the predicted texts.
    """
    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        dataset = [json.loads(line) for line in dataset_file if line.strip()]

    target_texts = [sample[target_field] for sample in dataset]
    pred_texts = [sample[pred_field] for sample in dataset]

    return target_texts, pred_texts


def evaluate_model(dataset_path: str | Path, target_field: str, pred_field: str) -> None:
    """
    Evaluate the model on a dataset.

    :param dataset_path: The path to the JSON Lines dataset.
    :param target_field: The name of the field containing the target texts.
    :param pred_field: The name of the field containing the predicted texts.
    """
    target_texts, pred_texts = load_target_pred_dataset(dataset_path, target_field, pred_field)

    bleu_score = get_bleu_score(target_texts, pred_texts)
    rouge_score = get_rouge_score(target_texts, pred_texts)
    bert_score = get_bert_score(target_texts, pred_texts)

    print(f"BLEU score: {bleu_score}")
    print(f"ROUGE score: {rouge_score}")
    print(f"Bert score: {bert_score}")

