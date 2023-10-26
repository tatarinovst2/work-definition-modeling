source config/common.sh
set -ex

echo -e '\n'

echo "Check requirements files"

configure_script

python config/stage_1_style_tests/requirements_check.py