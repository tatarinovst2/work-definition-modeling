#!/bin/bash
source config/common.sh

set -ex

echo -e '\n'
echo 'Running docstring style check...'

configure_script

pydocstyle --config ./config/docstrings_check/.pydocstyle config
darglint --config ./config/docstrings_check/.darglint config

directories=$(get_project_directories)

for directory in $directories; do
  pydocstyle --config ./config/docstrings_check/.pydocstyle "${directory}"
  darglint --config ./config/docstrings_check/.darglint "${directory}"
done

check_if_failed

echo "Docstring style check passed."
