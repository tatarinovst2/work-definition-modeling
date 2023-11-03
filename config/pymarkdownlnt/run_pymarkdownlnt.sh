#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running pymarkdownlnt check...'

configure_script

python3 -m pymarkdown --config config/pymarkdownlnt/.pymarkdownlnt.json scan *.md

directories=$(get_project_directories)

for directory in $directories; do
  python3 -m pymarkdown --config config/pymarkdownlnt/.pymarkdownlnt.json scan "${directory}"
done

check_if_failed

echo "Pymarkdownlnt check passed."
