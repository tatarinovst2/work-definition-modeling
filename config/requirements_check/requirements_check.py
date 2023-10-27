"""
Checks dependencies
"""
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


def get_paths() -> list[Path]:
    """
    Returns list of paths to non-python files
    """
    list_with_paths = []
    for file in ROOT_DIR.iterdir():
        if file.name in ['requirements.txt', 'requirements_ci.txt']:
            list_with_paths.append(file)
    return list_with_paths


def get_requirements(path: str | Path) -> list:
    """
    Returns a list of dependencies
    """
    with path.open(encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def compile_pattern() -> re.Pattern:
    """
    Returns the compiled pattern
    """
    return re.compile(r'\w+(-\w+|\[\w+\])*==\d+(\.\d+)+')


def check_dependencies(lines: list, compiled_pattern: re.Pattern) -> bool:
    """
    Checks that dependencies confirm to the template
    """
    if sorted(lines) != lines:
        print('Dependencies do not conform to the template.')
        return False
    for line in lines:
        if not re.search(compiled_pattern, line):
            print('Dependencies do not conform to the template.')
            return False
    print('Dependencies: OK.')
    return True


def main() -> None:
    """
    Calls functions
    """
    paths = get_paths()
    compiled_pattern = compile_pattern()

    for path in paths:
        print(f"Checking {path} file...")
        lines = get_requirements(path)
        if not check_dependencies(lines, compiled_pattern):
            sys.exit(1)


if __name__ == '__main__':
    main()
