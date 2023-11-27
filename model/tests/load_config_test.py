"""Tests for load_config function from utils.py"""
import unittest
from pathlib import Path

import pytest

from model.src.utils import load_train_config


class TestLoadConfig(unittest.TestCase):
    """Tests for load_train_config function from utils.py"""
    def setUp(self):
        self.tests_dir = Path(__file__).parent
        self.correct_config_path = self.tests_dir / "data" / "correct_train_config.json"
        self.incorrect_config_path = self.tests_dir / "data" / "incorrect_train_config.json"

    @pytest.mark.model
    def test_load_config_correct(self):
        """
        load_train_config should return the correct config
        """
        actual = load_train_config(self.correct_config_path)
        expected = {
            "model_checkpoint": "cointegrated/ruT5-small",
            "dataset_path": "wiktionary_parser/data/definitions.jsonl",
            "learning_rate": 2e-5,
            "batch_size": 4,
            "num_train_epochs": 1
        }

        self.assertEqual(actual, expected)

    @pytest.mark.model
    def test_load_config_incorrect(self):
        """
        load_train_config should raise a ValueError if the config is incorrect
        """
        with self.assertRaises(ValueError):
            load_train_config(self.incorrect_config_path)
