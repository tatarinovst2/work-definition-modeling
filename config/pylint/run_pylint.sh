#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running lint check...'

configure_script

python3 -m pylint --rcfile config/pylint/.pylintrc config

directories=$(get_project_directories)

for directory in $directories; do
  python3 -m pylint --rcfile config/pylint/.pylintrc "${directory}"
done

check_if_failed

echo "Lint check passed."
