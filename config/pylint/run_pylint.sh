#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running lint check...'

configure_script

export PYTHONPATH="${PYTHONPATH}:$(pwd)/config"
python3 -m pylint --rcfile config/pylint/.pylintrc config

directories=$(get_project_directories)

for directory in $directories; do
  export PYTHONPATH="${PYTHONPATH}:$(pwd)/${directory}"
  python3 -m pylint --rcfile config/pylint/.pylintrc "${directory}"
  check_if_failed
done

echo "Lint check passed."
