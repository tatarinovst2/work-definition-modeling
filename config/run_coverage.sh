#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running coverage check...'

configure_script

coverage run -m pytest "wiktionary_parser" "model_training"
coverage report
