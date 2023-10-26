#!/bin/bash
source config/common.sh

set -ex

echo -e '\n'
echo 'Running flake8 check...'

configure_script

python3 -m flake8 --config ./config/flake8/.flake8 *.py config wiktionary_parser model_training

check_if_failed

echo "Flake8 check passed."
