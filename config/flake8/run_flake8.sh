#!/bin/bash
source config/common.sh

set -ex

echo -e '\n'
echo 'Running flake8 check...'

configure_script

python3 -m flake8 --config ./config/flake8/.flake8 config

directories=$(get_project_directories)

for directory in $directories; do
  python3 -m flake8 --config ./config/flake8/.flake8 "${directory}"
done

check_if_failed

echo "Flake8 check passed."
