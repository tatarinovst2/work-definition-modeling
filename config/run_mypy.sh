#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running mypy check...'

configure_script

directories=$(get_project_directories)

python3 -m mypy "${directories}"

check_if_failed

echo "Mypy check passed."
