"""Tests for prepare_row and prepare_dataset functions."""
import unittest
from pathlib import Path

import pytest

from model_training.dataset_processing import prepare_dataset, prepare_row


class DatasetProcessingTest(unittest.TestCase):
    """
    Tests for functions that process the dataset
    """
    def setUp(self):
        self.tests_dir = Path(__file__).resolve().parent
        self.dataset_example_filepath = self.tests_dir / "data" / "dataset-sample.jsonl"

    @pytest.mark.model_training
    def test_prepare_row(self):
        """
        Prepare_row should return a dict with the correct keys and values
        """
        row = {"word": "слово", "definition": "значение", "context": "пример"}
        expected = {"input_text": "<LM>пример\n Определение слова \"слово\": ",
                    "target_text": "значение"}
        actual = prepare_row(row)
        self.assertEqual(expected, actual)

    @pytest.mark.model_training
    def test_prepare_dataset_splits(self):
        """
        Prepare_dataset should return a dataset with the correct splits
        """
        dataset = prepare_dataset(self.dataset_example_filepath)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset["train"]), 3)
        self.assertEqual(len(dataset["test"]), 1)

    @pytest.mark.model_training
    def test_prepare_dataset_content(self):
        """
        Prepare_dataset should return a dataset with the correct content
        """
        dataset = prepare_dataset(self.dataset_example_filepath)
        print(dataset["train"][0])
        print(dataset["test"][0])
        self.assertEqual(dataset["train"][0]["input_text"],
                         "<LM>Из кулинарных изысков английской кухни в список попали "
                         "сыр чеддер, ростбиф и йоркширский пудинг."
                         "\n Определение слова \"английский\": ")
        self.assertEqual(dataset["train"][0]["target_text"],
                         "относящийся к Англии, к англичанам или их языку; иногда "
                         "(ошибочно или расширительно) относящийся к Великобритании в целом")
        self.assertEqual(dataset["test"][0]["input_text"],
                         "<LM>Табличка на выезде была на английском, и гласила "
                         "«Don’t even think of parking here» вместо обычного "
                         "«Ausfahrt Tag und Nacht freihalten»."
                         "\n Определение слова \"английский\": ")
        self.assertEqual(dataset["test"][0]["target_text"],
                         "субстантивир., лингв. то же, что английский язык")

    @pytest.mark.model_training
    def test_prepare_dataset_with_invalid_filepath(self):
        """
        Prepare_dataset should raise a FileNotFoundError if the file doesn't exist
        """
        with self.assertRaises(FileNotFoundError):
            prepare_dataset("invalid_filepath")
