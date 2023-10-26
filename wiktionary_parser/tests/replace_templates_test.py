"""Tests for the calculate_precision function"""
import unittest

import pytest

from wiktionary_parser.wiktionary_parser import (
    replace_templates_in_definition, replace_templates_in_example)


class ReplaceTemplatesTest(unittest.TestCase):
    """
    Tests for the replace_templates_in_example and replace_templates_in_definition functions
    """

    @pytest.mark.wiktionary_parser
    def test_replace_templates_in_example_ideal(self):
        """
        Replace_templates_in_example should replace templates in example text with their values
        """
        example_text = "Вот {{выдел|это}} {{выдел|пример}} {{выдел|текста}}"
        expected_example_text = "Вот это пример текста"
        actual_example_text = replace_templates_in_example(example_text)

        self.assertEqual(expected_example_text, actual_example_text)

    @pytest.mark.wiktionary_parser
    def test_replace_templates_in_example_no_templates(self):
        """
        Replace_templates_in_example should not change the example text
        if it doesn't have any templates
        """
        example_text = "Вот это пример текста"
        expected_example_text = "Вот это пример текста"
        actual_example_text = replace_templates_in_example(example_text)

        self.assertEqual(expected_example_text, actual_example_text)

    @pytest.mark.wiktionary_parser
    def test_replace_templates_in_definition_ideal(self):
        """
        Replace_templates_in_definition should replace templates
        in definition text with their values
        """
        definition_text = ("{{устар.|ru}} вот это пример определения {{пример|{{выдел|Вот}} "
                           "{{выдел|это}}}}")
        expected_definition_text = "устар. вот это пример определения"
        actual_definition_text = replace_templates_in_definition(definition_text)

        self.assertEqual(expected_definition_text, actual_definition_text)

    @pytest.mark.wiktionary_parser
    def test_replace_templates_in_definition_no_templates(self):
        """
        Replace_templates_in_definition should not change the definition text
        if it doesn't have any templates
        """
        definition_text = "Вот это пример текста"
        expected_definition_text = "Вот это пример текста"
        actual_definition_text = replace_templates_in_definition(definition_text)

        self.assertEqual(expected_definition_text, actual_definition_text)
