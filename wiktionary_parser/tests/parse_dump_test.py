"""Tests for the parse_dump function"""
import json
import traceback
import unittest
from pathlib import Path

import pytest

from wiktionary_parser.wiktionary_parser import parse_dump


class ParseDumpTest(unittest.TestCase):
    """
    Tests for the parse_dump function
    """
    def setUp(self) -> None:
        self.test_data_dir = Path(__file__).resolve().parent / "data"
        self.test_articles_path = self.test_data_dir / "test-articles.xml.bz2"
        self.test_definitions_path = self.test_data_dir / "test-definitions.jsonl"

        self.expected_definitions_path = self.test_data_dir / "expected-output.jsonl"
        with open(self.expected_definitions_path, "r", encoding="utf-8") as file:
            self.expected_definitions = [json.loads(line) for line in file]

        if self.test_definitions_path.exists():
            self.test_definitions_path.unlink()

    @pytest.mark.wiktionary_parser
    def test_parse_dump_ideal(self):
        """
        Parse_dump should not raise any errors with ideal input
        """
        try:
            parse_dump(self.test_articles_path, self.test_definitions_path)
        except Exception:
            print(traceback.format_exc())
            self.fail("parse_dump raised an error unexpectedly!")

        self.assertTrue(self.test_definitions_path.exists())

        with open(self.test_definitions_path, "r", encoding="utf-8") as file:
            actual_definitions = [json.loads(line) for line in file]

        self.assertEqual(self.expected_definitions, actual_definitions)

    def tearDown(self):
        if self.test_definitions_path.exists():
            self.test_definitions_path.unlink()
