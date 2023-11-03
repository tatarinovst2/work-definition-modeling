#!/bin/bash
source config/common.sh

directories=$(get_project_directories)

echo ${directories}

echo "Running flake8 check..."
python3 -m flake8 --config ./config/flake8/.flake8 config

echo "Running pylint check..."
python3 -m pylint --rcfile config/pylint/.pylintrc config

echo "Running mypy check..."
python3 -m mypy config

for directory in $directories; do
  python3 -m flake8 --config ./config/flake8/.flake8 "${directory}"

  python3 -m pylint --rcfile config/pylint/.pylintrc "${directory}"

  python3 -m mypy ${directory}
done

echo "Running docstring style check..."
pydocstyle --config ./config/docstrings_check/.pydocstyle config
darglint -s sphinx -z short config

for directory in $directories; do
  pydocstyle --config ./config/docstrings_check/.pydocstyle "${directory}"
  darglint -s sphinx -z short "${directory}"
done

echo "Running pymarkdown check..."
python3 -m pymarkdown --config config/pymarkdownlnt/.pymarkdownlnt.json scan *.md

echo "Running spellcheck check..."
python3 -m pyspelling -c config/spellcheck/.spellcheck.yaml -v

echo "Running pytest..."
python3 -m pytest

echo "Running newline check..."
python3 config/newline_check/newline_check.py

echo "Running coverage check..."
coverage run -m pytest ${directories}
coverage report
