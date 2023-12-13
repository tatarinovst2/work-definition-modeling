"""Tests for the get_section function"""
import unittest
from pathlib import Path

import pytest

from wiktionary_parser.clean_dataset import clean_dataset, load_dataset, dump_dataset


class CleanDatasetTest(unittest.TestCase):
    """Tests for clean_dataset function"""
    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "data"
        self.dataset_to_clean = self.test_data_dir / "dataset_to_clean.jsonl"
        self.output_dataset = self.test_data_dir / "cleaned_test_dataset.jsonl"

    @pytest.mark.wiktionary_parser
    def test_load_dataset(self):
        """
        Load_dataset should return the dataset
        """
        actual = load_dataset(self.dataset_to_clean)
        self.assertEqual(len(actual), 4, "Should return the dataset")

    @pytest.mark.wiktionary_parser
    def test_clean_dataset(self):
        """
        Clean_dataset should return the cleaned dataset
        """
        dataset = load_dataset(self.dataset_to_clean)
        actual = clean_dataset(dataset)
        expected = [{'id': 167599, 'title': 'кабель', 'definitions': {
                         'один или несколько изолированных друг от друга проводников (жил), заключённых в оболочку':
                             {'examples': [
                                 'За восемь лет, прошедших до следующей экспедиции, было изготовлено более 30 000 км кабелей для 75 подводных линий.']
                             }, 'то же, что кабельное телевидение':
                             {'examples':
                                  ['Хичкоком, честно говоря, время от времени лишь балуюсь, когда его по кабелю показывают.']}}},
                    {'id': 167634, 'title': 'кабельное телевидение', 'definitions': {
                        'телевидение, в котором передача телевизионного сигнала осуществляется по специально проложенному кабелю':
                                         {'examples': []}}},
                    {'id': 167625, 'title': 'двояко', 'definitions': {
                        'двумя способами или путями, в двух видах, формах': {'examples': []}}
                     },
                    {'id': 167607, 'title': 'охрана', 'definitions': {
                        'действие по значению глагола защита чего-либо от посягательства':
                            {'examples': ['Серьёзной проблемой становится охрана коммуникаций.']},
                        'люди, техника и прочие ресурсы, предназначенные для охраны , а также службы, занимающиеся охраной':
                            {'examples': ['В городской вневедомственной охране впервые создаётся особое мобильное подразделение.']}}
                     }]
        print(actual)