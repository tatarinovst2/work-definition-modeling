#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running lint check...'

configure_script

python3 -m pylint --rcfile config/pylint/.pylintrc *.py config wiktionary_parser model_training

check_if_failed
