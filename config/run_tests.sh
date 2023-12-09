#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running pytest...'

configure_script

directories=$(get_project_directories)

for directory in $directories; do
  export PYTHONPATH="${PYTHONPATH}:$(pwd)/${directory}"
done

if [[ -z "${directories}" ]]; then
  echo "No tests to run currently."
  exit 0
fi

python3 -m pytest

check_if_failed

echo "Pytest passed."
