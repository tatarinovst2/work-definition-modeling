"""Tests for parse_wiki function"""
import unittest
from pathlib import Path
from unittest import mock

import pytest

from wiktionary_parser.run_parser import parse_wiki
from wiktionary_parser.template_parsing import load_config


class ParseWikiTest(unittest.TestCase):
    """Tests for parse_wiki function"""
    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "data"
        self.test_config_path = self.test_data_dir / "test_parser_config.json"

        self.config = load_config(self.test_config_path)

        self.wiki_ideal = (self.test_data_dir / "wiki_ideal.txt").open(
            "r", encoding="utf-8").read()

        self.wiki_multiple_articles_per_word = (self.test_data_dir /
                                                "wiki_multiple_articles_per_word.txt").open(
            "r", encoding="utf-8").read()

        self.wiki_en = (self.test_data_dir / "wiki_en.txt").open(
            "r", encoding="utf-8").read()

        self.wiki_no_definitions_section = (self.test_data_dir /
                                            "wiki_no_definitions_section.txt").open(
            "r", encoding="utf-8").read()


    @pytest.mark.wiktionary_parser
    def test_parse_wiki_ideal(self):
        """
        Parse_wiki should not raise any errors with ideal input
        """
        try:
            actual = parse_wiki(self.wiki_ideal, self.config)
        except Exception:
            self.fail("parse_wiki raised Exception unexpectedly!")

        expected = {'составляющий четвертую часть (четверть) от чего либо':
                        {'examples':
                             ['Таким образом, к малоформатным печатным машинам можно отнести '
                              'машины «четвертинных» форматов А3-В3.']},
                    'дважды свёрнутый':
                        {'examples':
                             ['От теста отщипываем по небольшому кусочку, раскатываем из него '
                              'кружок потоньше, толщиной примерно 3 мм., обмакиваем его одной '
                              'стороной в тарелку с сахаром, складываем вдвое сахаром внутрь, '
                              'снова обмакиваем и складываем по такому же принципу. Полученный '
                              'в итоге четвертинный конвертик одной стороной снова обмакиваем '
                              'в сахар и кладём на противень противоположной.']
                         }
                    }

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_parse_wiki_multiple_articles_per_word(self):
        """
        Parse_wiki should return all definitions for a word
        """
        actual = parse_wiki(self.wiki_multiple_articles_per_word, self.config)
        expected = {'твёрдый стержень с особым сочетанием выступов и углублений для запирания '
                    'и отпирания замко́в':
                        {'examples':
                             ['Ключ от парадной двери.', 'Ключ от сарая.', 'Дубликат ключей.',
                              'Он лежал в первой комнате на постели, подложив одну руку под '
                              'затылок, а другой держа погасшую трубку; дверь во вторую '
                              'комнату была заперта на замок, и ключа в замке не было.',
                              'В саду, за малиной, есть калитка, её маменька запирает на '
                              'замок, а ключ прячет.',
                              'Не поднимая никакого шума, доктор отпер дверь своим ключом и, '
                              'войдя, тотчас запер за собою двери и не вынул ключа, так, чтобы '
                              'уже ещё никто не мог отпереть её, а должен был бы постучаться.']
                         },
                    'инструмент для завинчивания и отвинчивания, приведения в действие '
                    'механизмов и других механических операций':
                        {'examples':
                             ['Гаечный ключ.', 'Разводной ключ.',
                              'Шофер повернул ключ зажигания, рывком, со звоном включил скорость, '
                              'и машина тронулась.']
                         },
                    'естественный источник воды, место выхода подземных вод на поверхность; '
                    'источник, родник':
                        {'examples':
                             ['В лесу бьёт ключ.', 'Напиться из ключа.',
                              'Студеный ключ играет по оврагу.']
                         }
                    }

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_parse_wiki_en_text(self):
        """
        Parse_wiki should not find anything here
        """
        actual = parse_wiki(self.wiki_en, self.config)
        expected = None

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_parse_wiki_no_definitions_section(self):
        """
        Parse_wiki should not find anything here
        """
        actual = parse_wiki(self.wiki_no_definitions_section, self.config)
        expected = {}

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_parse_wiki_empty_definitions(self):
        """
        Parse_wiki should not find anything here
        """
        with mock.patch("wiktionary_parser.run_parser.pop_templates_in_text",
                        return_value=("", [])):
            actual = parse_wiki(self.wiki_ideal, self.config)
        expected = {}

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_parse_wiki_empty_examples(self):
        """
        Parse_wiki should not find anything here
        """
        with mock.patch("wiktionary_parser.run_parser.replace_templates_with_text",
                        return_value=""):
            actual = parse_wiki(self.wiki_ideal, self.config)
        expected = {}

        self.assertEqual(expected, actual)

    @pytest.mark.wiktionary_parser
    def test_parse_wiki_no_lists_found(self):
        """
        Parse_wiki should not find anything here
        """
        with mock.patch("wikitextparser.WikiText.get_lists", return_value=[]):
            actual = parse_wiki(self.wiki_ideal, self.config)
            expected = {}

            self.assertEqual(expected, actual)
