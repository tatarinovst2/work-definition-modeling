"""Tests for the get_section function"""
import json
import unittest
from pathlib import Path

import pytest

from wiktionary_parser.clean_dataset import clean_dataset, dump_dataset, load_dataset


class CleanDatasetTest(unittest.TestCase):
    """Tests for clean_dataset function"""
    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "data"
        self.dataset_to_clean = self.test_data_dir / "dataset_to_clean.jsonl"
        self.dataset_expected = self.test_data_dir / "dataset_expected_after_cleaning.json"
        self.output_dataset = self.test_data_dir / "dumped_dataset.jsonl"

        if self.output_dataset.exists():
            self.output_dataset.unlink()

    @pytest.mark.wiktionary_parser
    def test_load_dataset(self):
        """
        Load_dataset should return the dataset
        """
        actual = load_dataset(self.dataset_to_clean)
        self.assertEqual(len(actual), 4, "Should return the dataset")

    @pytest.mark.wiktionary_parser
    def test_clean_dataset(self):
        """
        Clean_dataset should return the cleaned dataset
        """
        dataset = load_dataset(self.dataset_to_clean)
        actual = clean_dataset(dataset)
        expected = json.loads(self.dataset_expected.open("r", encoding="utf-8").read())
        print(actual)
        self.assertEqual(expected, actual, "Should return the cleaned dataset")

    @pytest.mark.wiktionary_parser
    def test_dump_dataset(self):
        """
        Dump_dataset should dump the dataset to the given path
        """
        dataset = load_dataset(self.dataset_to_clean)
        cleaned_dataset = clean_dataset(dataset)
        dump_dataset(cleaned_dataset, self.output_dataset)
        actual = load_dataset(self.output_dataset)
        expected = json.loads(self.dataset_expected.open("r", encoding="utf-8").read())
        self.assertEqual(expected, actual, "Should dump the dataset to the given path")

    def tearDown(self):
        if self.output_dataset.exists():
            self.output_dataset.unlink()
