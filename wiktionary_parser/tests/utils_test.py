"""Tests for utils module."""
import unittest

import pytest

from wiktionary_parser.utils import clean_text, remove_text_before_words


class UtilsTest(unittest.TestCase):
    """Tests for utils module"""

    @pytest.mark.wiktionary_parser
    def test_clean_text_ideal(self):
        """Clean_text should clean the text"""
        text = "определение= , также meaningful text"
        expected = "meaningful text"
        actual = clean_text(text, ["также"], ["определение="])

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_clean_text_no_words_to_remove(self):
        """Clean_text should clean the text even if no words to remove are specified"""
        text = " , также meaningful text"
        expected = "также meaningful text"
        actual = clean_text(text)

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_clean_text_wiki_numbers(self):
        """Clean_text should remove wiki numbers"""
        text = "[1] meaningful text"
        expected = "meaningful text"
        actual = clean_text(text)

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_remove_text_before_words_ideal(self):
        """Remove_text_before_words should remove text before the first word in a text"""
        text = " , также meaningful text"
        expected = "также meaningful text"
        actual = remove_text_before_words(text)

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_remove_text_before_words_no_text_to_remove(self):
        """Remove_text_before_words should return the text if there is no text to remove"""
        text = "meaningful text"
        expected = "meaningful text"
        actual = remove_text_before_words(text)

        self.assertEqual(expected, actual)
