"""Tests for the validate_filepaths function"""
import traceback
import unittest
from pathlib import Path

import pytest

from wiktionary_parser.run_parser import validate_filepaths


class ValidateFilepathsTest(unittest.TestCase):
    """
    Tests for the validate_filepaths function
    """
    def setUp(self) -> None:
        self.test_data_dir = Path(__file__).resolve().parent / "data"
        self.test_articles_path = self.test_data_dir / "test_articles.xml.bz2"
        self.test_definitions_path = self.test_data_dir / "test-definitions.jsonl"

    @pytest.mark.wiktionary_parser
    def test_validate_filepaths_ideal(self):
        """
        Validate_filepaths should not raise any errors with ideal input
        """
        try:
            validate_filepaths(self.test_articles_path, self.test_definitions_path)
        except FileNotFoundError:
            print(traceback.format_exc())
            self.fail("validate_filepaths raised FileNotFoundError unexpectedly!")

    @pytest.mark.wiktionary_parser
    def test_validate_filepaths_missing_input(self):
        """
        Validate_filepaths should raise FileNotFoundError if input file is missing
        """
        with self.assertRaises(FileNotFoundError):
            validate_filepaths(self.test_data_dir / "missing-input.xml.bz2",
                               self.test_definitions_path)

    @pytest.mark.wiktionary_parser
    def test_validate_filepaths_missing_output(self):
        """
        Validate_filepaths should create output directory if it doesn't exist
        """
        try:
            validate_filepaths(self.test_articles_path,
                               self.test_data_dir / "missing-directory" / "missing-output.jsonl")
        except FileNotFoundError:
            print(traceback.format_exc())
            self.fail("validate_filepaths raised FileNotFoundError unexpectedly!")

        self.assertTrue((self.test_data_dir / "missing-directory").exists())

        (self.test_data_dir / "missing-directory").rmdir()

    @pytest.mark.wiktionary_parser
    def test_validate_filepaths_existing_output(self):
        """
        Validate_filepaths should delete existing output file
        """
        self.test_definitions_path.touch()
        try:
            validate_filepaths(self.test_articles_path, self.test_definitions_path)
        except FileNotFoundError:
            print(traceback.format_exc())
            self.fail("validate_filepaths raised FileNotFoundError unexpectedly!")
        self.assertFalse(self.test_definitions_path.exists())
