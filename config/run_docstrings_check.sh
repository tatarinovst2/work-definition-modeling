#!/bin/bash
source config/common.sh

set -ex

echo -e '\n'
echo 'Running docstring style check...'

configure_script

pydocstyle config
darglint -s sphinx -z short config

directories=$(get_project_directories)

for directory in $directories; do
  pydocstyle "${directory}"
  check_if_failed

  darglint -s sphinx -z short "${directory}"
  check_if_failed
done

echo "Docstring style check passed."
