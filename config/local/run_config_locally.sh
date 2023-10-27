#!/bin/bash

echo "Running flake8 check..."
python3 -m flake8 --config ./config/flake8/.flake8 config wiktionary_parser model_training

echo "Running pylint check..."
python3 -m pylint --rcfile config/pylint/.pylintrc config wiktionary_parser model_training

echo "Running pymarkdown check..."
python3 -m pymarkdown --config config/pymarkdownlnt/.pymarkdownlnt.json scan *.md

echo "Running spellcheck check..."
python3 -m pyspelling -c config/spellcheck/.spellcheck.yaml -v

echo "Running mypy check..."
python3 -m mypy wiktionary_parser model_training

echo "Running pytest..."
python3 -m pytest

echo "Running coverage check..."
coverage run -m pytest "wiktionary_parser" "model_training"
coverage report
