import re

from bs4 import BeautifulSoup
import requests

from utils import ResponseCode, STATUSES


class WiktionaryParser:
    MISSING_SYMBOL = "—"
    SEMANTICS_SPLIT_SYMBOL = "◆"
    SEMANTICS_ID = "Семантические_свойства"
    MORPHOLOGY_ID = "Морфологические_и_синтаксические_свойства"

    def __init__(self) -> None:
        self.result = dict()
        self.url = "https://ru.wiktionary.org/w/index.php?title={}"
        self.session = requests.Session()

    def _set_definitions(self, soup: BeautifulSoup) -> None:
        self.result["definitions"] = list()
        semantic_title = soup.find(id=self.SEMANTICS_ID)
        definition_text = semantic_title.find_next("ol")

        for item in definition_text.find_all("li"):
            text = item.get_text()
            if text:
                text = "".join(re.findall("[-◆А-яё.,! ]+", text))
                text_split = text.split(self.SEMANTICS_SPLIT_SYMBOL)
                self.result["definitions"].append({
                    "value": text_split[0].strip(),
                    "example": text_split[1].strip()
                })

    def process_html_page(self, page: str) -> dict:
        soup = BeautifulSoup(page, features="html.parser")
        self._set_definitions(soup)
        return self.result

    def make_request(self, word: str):
        response = self.session.get(self.url.format(word))  # Use the session for the request
        if response.status_code == ResponseCode.SUCCESS.value:
            return {**self.process_html_page(response.text), **STATUSES[response.status_code]}
        # Handle other status codes as needed
        self.result = {**self.result, **STATUSES[response.status_code]}
        return self.result
