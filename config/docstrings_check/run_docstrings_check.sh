#!/bin/bash
source config/common.sh

set -ex

echo -e '\n'
echo 'Running docstring style check...'

configure_script

pydocstyle --config ./config/docstrings_check/.pydocstyle config wiktionary_parser model_training

darglint --config ./config/docstrings_check/.darglint config wiktionary_parser model_training

check_if_failed

echo "Docstring style check passed."
