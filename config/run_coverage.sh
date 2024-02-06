#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running coverage check...'

configure_script

directories=$(get_project_directories)
INITIAL_PYTHONPATH=PYTHONPATH

if [[ -z "${directories}" ]]; then
  echo "No files to run coverage on currently."
  exit 0
fi

for directory in $directories; do
  export PYTHONPATH="${INITIAL_PYTHONPATH}:$(pwd)/${directory}"
  coverage run --append -m pytest "$directory"
done

coverage report

check_if_failed
