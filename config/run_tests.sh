#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running pytest...'

configure_script

python3 -m pytest

check_if_failed

echo "Pytest passed."
