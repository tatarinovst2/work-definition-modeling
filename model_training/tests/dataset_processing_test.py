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
        self.validation_dataset_filepath = self.tests_dir / "data" / "validation-dataset.jsonl"

        if self.validation_dataset_filepath.exists():
            self.validation_dataset_filepath.unlink()

    @pytest.mark.model_training
    def test_prepare_row(self):
        """
        Prepare_row should return a dict with the correct keys and values
        """
        row = {"word": "слово", "definition": "значение", "context": "пример"}
        expected = {"input_text": "<LM>Контекст: \"пример\" Определение слова \"слово\": ",
                    "target_text": "значение"}
        actual = prepare_row(row)
        self.assertEqual(expected, actual)

    @pytest.mark.model_training
    def test_prepare_dataset_splits(self):
        """
        Prepare_dataset should return a dataset with the correct splits
        """
        dataset = prepare_dataset(self.dataset_example_filepath)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(dataset["train"]), 3)
        self.assertEqual(len(dataset["test"]), 1)
        self.assertEqual(len(dataset["validation"]), 1)

    @pytest.mark.model_training
    def test_prepare_dataset_content(self):
        """
        Prepare_dataset should return a dataset with the correct content
        """
        dataset = prepare_dataset(self.dataset_example_filepath)
        self.assertEqual(dataset["train"][0]["input_text"],
                         "<LM>Контекст: \"Табличка на выезде была на английском, и гласила "
                         "«Don’t even think of parking here» вместо обычного «Ausfahrt Tag und "
                         "Nacht freihalten».\" Определение слова \"английский\": ")
        self.assertEqual(dataset["train"][0]["target_text"],
                         "то же, что английский язык")
        self.assertEqual(dataset["test"][0]["input_text"],
                         "<LM>Контекст: \"У неё были две короткохвостые шиншиллы: одна "
                         "эбонитовая, а другая — бежевая.\" Определение слова \"эбонитовый\": ")
        self.assertEqual(dataset["test"][0]["target_text"],
                         "очень чёрный")
        self.assertEqual(dataset["validation"][0]["input_text"],
                         "<LM>Контекст: \"Из кулинарных изысков английской кухни в "
                         "список попали сыр чеддер, ростбиф и йоркширский пудинг.\" "
                         "Определение слова \"английский\": ")
        self.assertEqual(dataset["validation"][0]["target_text"],
                        "относящийся к Англии, к англичанам или их языку; иногда (ошибочно "
                        "или расширительно) относящийся к Великобритании в целом")

    @pytest.mark.model_training
    def test_prepare_dataset_save_validation(self):
        """
        Prepare_dataset should save the validation dataset to a file if the filepath is given
        """
        dataset = prepare_dataset(self.dataset_example_filepath,
                                  test_dataset_output_path=self.validation_dataset_filepath)
        self.assertTrue(self.validation_dataset_filepath.exists())
        self.assertEqual(len(dataset["validation"]), 1)

    @pytest.mark.model_training
    def test_prepare_dataset_with_invalid_filepath(self):
        """
        Prepare_dataset should raise a FileNotFoundError if the file doesn't exist
        """
        with self.assertRaises(FileNotFoundError):
            prepare_dataset("invalid_filepath")

    def tearDown(self):
        if self.validation_dataset_filepath.exists():
            self.validation_dataset_filepath.unlink()
