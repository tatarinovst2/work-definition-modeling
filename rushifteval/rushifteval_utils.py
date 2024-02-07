"""Utility module for rushifteval."""
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic.dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances
from sklearn.preprocessing import normalize

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class AnnotatedWord:
    """
    A dataclass representing a row in the TSV dataset.

    :param word: The target word.
    :param sent1: The first sentence.
    :param sent2: The second sentence.
    :param mean: The mean annotation value.
    :param vect1: Embedding for the usage in the first sentence.
    :param vect2: Embedding for the usage in the second sentence.
    """

    word: str
    sent1: str
    sent2: str
    mean: float
    vect1: Optional[list[float]] = None
    vect2: Optional[list[float]] = None


def load_jsonl_vectors(jsonl_file_path: str | Path) -> dict[tuple[int, int], list[float]]:
    """
    Load vectors from a JSONL file.

    :param jsonl_file_path: Path to the JSONL file containing sentence vectors.
    :return: A dictionary with keys as (id, sentence_id) and values as vectors.
    """
    vectors = {}
    with open(jsonl_file_path, 'r', encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            key = (data['id'], data['sentence_id'])
            vectors[key] = data['vector']
    return vectors


def compute_distance(vect1: list[float], vect2: list[float], metric: str,
                     normalize_flag: bool) -> float:
    """
    Compute distance between two vectors using the specified metric.

    :param vect1: The first vector.
    :param vect2: The second vector.
    :param metric: The metric to use for distance computation.
    :param normalize_flag: Whether to normalize the vectors before computing distance.
    :raises ValueError: If given an empty value.
    :return: The computed distance between the two vectors.
    """
    if vect1 is None or vect2 is None:
        raise ValueError("Vector is None!")

    if normalize_flag:
        vect1 = normalize([vect1])[0]
        vect2 = normalize([vect2])[0]

    if metric == 'cosine':
        return cosine_distances([vect1], [vect2])[0][0]

    if metric == 'manhattan':
        return manhattan_distances([vect1], [vect2])[0][0]

    return float(np.linalg.norm(np.array(vect1) - np.array(vect2)))


def load_annotated_data(tsv_file_path: str | Path,
                        jsonl_vectors: dict[tuple[int, int], list[float]]) -> list[AnnotatedWord]:
    """
    Load annotated data from a TSV file.

    :param tsv_file_path: Path to the TSV file containing annotations.
    :param jsonl_vectors: Dictionary with loaded vectors from the JSONL file.
    :return: A list of AnnotatedWord instances.
    """
    annotated_data = []
    with open(tsv_file_path, 'r', encoding="utf-8") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        for idx, row in enumerate(reader):
            row_id = idx
            word = row['word']
            sent1 = row['sent1']
            sent2 = row['sent2']
            mean = float(row['mean'])
            vect1 = jsonl_vectors.get((row_id, 1))
            vect2 = jsonl_vectors.get((row_id, 2))
            annotated_data.append(
                AnnotatedWord(word=word,
                              sent1=sent1,
                              sent2=sent2,
                              mean=mean,
                              vect1=vect1,
                              vect2=vect2)
            )
    return annotated_data


def parse_path(path: str | Path) -> Path:
    """
    Ensure that the path is absolute and is in a pathlib.Path format.

    :param path: The path to parse.
    :return: The parsed path.
    """
    path = Path(path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path
