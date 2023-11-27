"""Tests for the get_section function"""
import unittest
from pathlib import Path

import pytest
import wikitextparser as wtp

from wiktionary_parser.run_parser import get_sections


class GetSectionTest(unittest.TestCase):
    """Tests for get_section function"""
    def setUp(self):
        self.test_data_dir = Path(__file__).resolve().parent / "data"
        with open(self.test_data_dir / "section_ideal.txt", "r", encoding="utf-8") as file:
            self.section_ideal = file.read()

    @pytest.mark.wiktionary_parser
    def test_get_section_ideal(self):
        """
        Get_section should return the section if it exists
        """
        wiki_text = wtp.parse(self.section_ideal)

        actual = get_sections(wiki_text, "Значение")
        self.assertEqual(len(actual), 1, "Should return one section (Значение)")

        expected = wiki_text.get_sections()
        self.assertEqual(expected[2].string, actual[0].string,
                         "Should return the correct section")

    @pytest.mark.wiktionary_parser
    def test_get_section_no_section(self):
        """
        Get_section should return an empty list if the section doesn't exist
        """
        wiki_text = wtp.parse("""=== Семантические свойства ===""")
        actual = get_sections(wiki_text, "Значение")
        self.assertEqual(len(actual), 0, "Should return an empty list")
