#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running lint check...'

configure_script

python3 -m pylint --rcfile config/pylint/.pylintrc config

directories=$(get_project_directories)

for directory in $directories; do
  export PYTHONPATH=$(pwd)/$directory:$PYTHONPATH
  python3 -m pylint --rcfile config/pylint/.pylintrc "${directory}"
  check_if_failed
done

echo "Lint check passed."
