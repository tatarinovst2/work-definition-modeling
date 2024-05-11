"""Tests for testing vectorize module."""
import json
import unittest
from pathlib import Path

import pytest

from vizvector.vectorize import load_json_dataset, save_json_dataset, vectorize_text


class TestVectorizationProcess(unittest.TestCase):
    """Tests for testing vectorize module."""

    def setUp(self):
        self.input_file_path = Path(__file__).parent / "data" / "preds.jsonl"
        self.output_file_path = Path(__file__).parent / "data" / "vectors.jsonl"
        self.expected_file_path = Path(__file__).parent / "data" / "expected_vectors.jsonl"
        self.text_column = "generated_text"
        self.model_name = "cointegrated/rubert-tiny2"

    @pytest.mark.vizvector
    def test_vectorization_process(self):
        """Test functions from vectorize module."""
        data = load_json_dataset(self.input_file_path)

        self.assertEqual(len(data), 5)
        self.assertEqual(data[0]['generated_text'], "значение, влияние кого-, чего-либо")

        vectorized_data = vectorize_text(data, self.text_column, self.model_name)

        save_json_dataset(vectorized_data, self.output_file_path)

        self.assertTrue(self.output_file_path.exists())

        with open(self.output_file_path, 'r', encoding='utf-8') as actual_file:
            actual = [json.loads(line) for line in actual_file]

        with open(self.expected_file_path, "r", encoding="utf-8") as expected_file:
            expected = [json.loads(line) for line in expected_file]

        self.assertEqual(actual, expected)

    def tearDown(self):
        if self.output_file_path.exists():
            self.output_file_path.unlink()
