#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running pytest...'

configure_script

directories=$(get_project_directories)

if [[ -z "${directories}" ]]; then
  echo "No tests to run currently."
  exit 0
fi

python3 -m pytest

check_if_failed

echo "Pytest passed."
