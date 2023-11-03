#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running mypy check...'

configure_script

python3 -m mypy wiktionary_parser model_training

check_if_failed

echo "Mypy check passed."
