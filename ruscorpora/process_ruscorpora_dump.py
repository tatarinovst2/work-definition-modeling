"""Process the dump of the Russian National Corpus to extract word usages."""
import gzip
import json

from pydantic.dataclasses import dataclass
from pymorphy3 import MorphAnalyzer
from razdel import tokenize
from tqdm import tqdm

from ruscorpora_utils import parse_path, ROOT_DIR, write_results_to_file


@dataclass
class Epoch:
    """
    Dataclass representing an epoch in the Russian National Corpus dump.

    :param date: The date of the epoch.
    :param file_path: The path to the file containing the epoch data.
    """

    date: str
    file_path: str


@dataclass
class ProcessRuscorporaDumpConfig:
    """
    Dataclass representing the configuration for processing the Russian National Corpus dump.

    :param epochs: List of Epoch instances.
    :param words: List of words to search for.
    :param output_file_path: Path to the output file.
    """

    epochs: list[Epoch]
    words: list[str]
    output_file_path: str


config_file_path = ROOT_DIR / "ruscorpora" / "process_ruscorpora_config.json"


def load_config(config_path: str) -> ProcessRuscorporaDumpConfig:
    """
    Load the configuration file.

    :param config_path: Path to the configuration file.
    :return: Configuration as a SampleCorporaConfig instance.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config_dict = json.load(file)
    epochs = [Epoch(**epoch) for epoch in config_dict["epochs"]]
    return ProcessRuscorporaDumpConfig(epochs=epochs, words=config_dict["words"],
                                       output_file_path=config_dict["output_file_path"])


def get_word_forms(word: str) -> list[str]:
    """
    Generate all possible forms of the word using pymorphy3.

    :param word: The word to generate forms for.
    :return: List of all possible word forms.
    """
    morph = MorphAnalyzer()
    parsed_word = morph.parse(word)
    forms = set()
    for p in parsed_word:
        forms.update({form.word for form in p.lexeme})
    return list(forms)


def process_epochs(words: list[str], epochs: list[Epoch]) -> list[dict]:
    """
    Process each epoch for each word, searching sentences and constructing result entries.

    :param words: List of words to search for.
    :param epochs: List of epoch instances, each containing date and file path.
    :return: List of dictionaries, each representing a found instance with details.
    """
    results = []
    id_counter = 1

    word_forms_dict = {word: get_word_forms(word) for word in words}

    for epoch in epochs:
        print(f"Checking epoch {epoch.date}.")
        with gzip.open(parse_path(epoch.file_path), 'rt', encoding='utf-8') as file:
            for line in tqdm(file):
                sentence = line.strip()
                tokens = [token.text.lower() for token in tokenize(sentence)]
                for word, word_forms in word_forms_dict.items():
                    if any(token.lower() in tokens for token in word_forms):
                        result = {
                            "id": id_counter,
                            "word": word,
                            "date": epoch.date,
                            "sentence": sentence,
                            "input_text": f'<LM> Контекст: "{sentence}" '
                                          f'Определение слова "{word}": '
                        }
                        results.append(result)
                        id_counter += 1
    return results


def main() -> None:
    """Process ruscorpora dump to extract word usages."""
    config = load_config(config_file_path)

    results = process_epochs(config.words, config.epochs)

    output_path = parse_path(config.output_file_path)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    write_results_to_file(results, output_file=output_path)


if __name__ == "__main__":
    main()