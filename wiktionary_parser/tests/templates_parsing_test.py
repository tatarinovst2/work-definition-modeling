"""Tests for the templates_parsing module"""
import unittest

import pytest

from wiktionary_parser.template_parsing import (TemplateMapping, get_templates,
                                                pop_templates_in_text, replace_templates_with_text,
                                                Template)


class TemplateParsingTest(unittest.TestCase):
    """Tests for the templates_parsing module"""

    def setUp(self):
        self.custom_mappings = [
            TemplateMapping(title_index=3, title="non-standard1", description_indexes=[0, 1, 3, 4],
                            arguments_count=5, starting_text="", ending_text=""),
            TemplateMapping(title_index=0, title="non-standard2", description_indexes=[0, 1, 3, 4],
                            arguments_count=-1, starting_text="", ending_text=""),
            TemplateMapping(title_index=0, title="non-standard3", description_indexes=[1],
                            arguments_count=7, starting_text="", ending_text=""),
            TemplateMapping(title_index=0, title="действие", description_indexes=[2],
                            arguments_count=3, starting_text="действие по значению глагола ",
                            ending_text=""),
            TemplateMapping(title_index=1, title="ru", description_indexes=[0], arguments_count=2,
                            starting_text="", ending_text="")
        ]

    @pytest.mark.wiktionary_parser
    def test_get_templates_ideal(self):
        """Get_templates should return a list of templates"""
        text = "Вот {{выдел|это}} {{выдел|пример}} {{выдел|текста}}"
        expected_templates = ["выдел", "выдел", "выдел"]
        actual_templates = [template.get_title() for template in get_templates(text)]

        self.assertEqual(expected_templates, actual_templates)

    @pytest.mark.wiktionary_parser
    def test_get_templates_no_templates(self):
        """Get_templates should return an empty list if the text doesn't have any templates"""
        text = "Вот это пример текста"
        expected_templates = []
        actual_templates = get_templates(text)

        self.assertEqual(expected_templates, actual_templates)

    @pytest.mark.wiktionary_parser
    def test_create_template_ideal(self):
        """Create a Template object"""
        template = Template(2, "выдел|это", ["выдел", "это"])

        self.assertEqual(2, template.level)
        self.assertEqual("выдел|это", template.raw_text)
        self.assertEqual(["выдел", "это"], template.arguments)

        self.assertEqual("выдел", template.get_title())
        self.assertEqual("это", template.get_text_from_argument_indexes([1],
                                                                        ", "))

    @pytest.mark.wiktionary_parser
    def test_create_template_representation(self):
        """Create a Template object with correct __str__ and __repr__ methods"""
        template = Template(2, "выдел|это", ["выдел", "это"])

        expected = ("Template(level=2, raw_text=выдел|это, "
                    "arguments=['выдел', 'это']")
        self.assertEqual(str(template), expected)
        self.assertEqual(repr(template), expected)

    @pytest.mark.wiktionary_parser
    def test_template_get_text_with_custom_mapping(self):
        """Template.get_text should return the text of the template with the custom mapping"""
        template = Template(2, "разг.|ru", ["разг.", "ru"])

        expected = "разг."
        actual = template.get_text(mappings=self.custom_mappings)

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_template_get_text_no_mappings(self):
        """Template.get_text should return the text of the template"""
        template = Template(2, "выдел|это", ["выдел", "это"])

        expected = "это"
        actual = template.get_text()

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_template_get_text_one_argument(self):
        """Template.get_text should return the text of the template"""
        template = Template(2, "выдел", ["выдел"])

        expected = "выдел"
        actual = template.get_text()

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_pop_templates_in_text_ideal(self):
        """Pop_templates_in_text should return the text without the templates, then – templates"""
        text = "Вот {{выдел|это}} {{выдел|пример}} {{выдел|текста}}"
        expected_text = "Вот"
        expected_templates = ["выдел|это", "выдел|пример", "выдел|текста"]
        actual_text, actual_templates = pop_templates_in_text(text, "выдел")

        self.assertEqual(expected_text, actual_text.strip())

        for index, template in enumerate(actual_templates):
            self.assertEqual(expected_templates[index], template.raw_text)

    @pytest.mark.wiktionary_parser
    def test_pop_templates_in_text_with_mappings(self):
        """Pop_templates_in_text should return the text without the templates, then – templates"""
        text = ("{{разг.|ru}} лицо мужского пола "
                "{{пример|Мужчины бились об заклад, кого родит графиня}}")
        expected_text = "лицо мужского пола {{пример|Мужчины бились об заклад, кого родит графиня}}"
        expected_templates = ["разг.|ru"]

        actual_text, actual_templates = pop_templates_in_text(text, "ru",
                                                              mappings=self.custom_mappings)

        self.assertEqual(expected_text, actual_text.strip())

        for index, template in enumerate(actual_templates):
            self.assertEqual(expected_templates[index], template.raw_text)

    @pytest.mark.wiktionary_parser
    def test_replace_templates_in_text_ideal(self):
        """Replace_templates_in_text should replace the templates in the text"""
        text = "Вот {{выдел|это}} {{выдел|пример}} {{выдел|текста}}"
        expected = "Вот"
        actual = replace_templates_with_text(text, templates_to_remove=["выдел"])

        self.assertEqual(expected, actual.strip())

    @pytest.mark.wiktionary_parser
    def test_replace_templates_in_text_with_mappings(self):
        """Replace_templates_in_text should replace the templates in the text"""
        text = "{{разг.|ru}} лицо мужского пола"
        expected = "разг. лицо мужского пола"
        actual = replace_templates_with_text(text, mappings=self.custom_mappings)

        self.assertEqual(expected, actual.strip())
