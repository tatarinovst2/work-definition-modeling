"""Module for extra functions that are used in the parser."""
import re


def clean_text(text: str, words_to_remove: list[str] | None = None,
               sequences_to_remove: list[str] | None = None) -> str:
    """
    Clean text by removing leading spaces, punctuation and specified words.

    :param text: The input text to be cleaned.
    :param words_to_remove: (optional) A list of words to remove from the beginning of the text.
    :param sequences_to_remove: (optional) A list of sequences to remove from the whole text.
    :return: The cleaned text.

    :Example:

    >>> clean_text(" , также meaningful text", ["также"])
    'meaningful text'
    """
    if words_to_remove is None:
        words_to_remove = []

    if sequences_to_remove is None:
        sequences_to_remove = []

    for sequence in sequences_to_remove:
        text = text.replace(sequence, " ")

    text = remove_text_before_words(text)
    text = re.sub(r"\[\d+]", "", text)

    for word in words_to_remove:
        text = text.strip()
        if text.startswith(word) and len(word) != len(text) and text[len(word)] in [",", " "]:
            text = text[len(word):]
            text = remove_text_before_words(text)

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def remove_text_before_words(text: str) -> str:
    """
    Remove punctuation and spaces before the first word in a text.

    :param text: The input text to be cleaned.
    :return: The cleaned text.
    """
    index = 0
    while index < len(text) and not text[index].isalnum() and text[index] != "[":
        index += 1
    return text[index:]
