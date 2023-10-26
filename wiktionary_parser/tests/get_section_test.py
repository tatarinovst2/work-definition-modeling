"""Tests for the get_section function"""
import unittest

import pytest
import wikitextparser as wtp

from wiktionary_parser.wiktionary_parser import get_sections


class GetSectionTest(unittest.TestCase):
    """Tests for get_section function"""
    @pytest.mark.wiktionary_parser
    def test_get_section_ideal(self):
        """
        Get_section should return the section if it exists
        """
        wiki_text = wtp.parse("""=== Семантические свойства ===

==== Значение ====
# [[сенат]] (орган государственной власти в разных странах или часть такого органа 
(палата парламента)) {{пример|US Senate is the upper house of the US Congress|
перевод=Сенат США- верхняя палата Конгресса США}}

==== Синонимы ===="""
        )

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
