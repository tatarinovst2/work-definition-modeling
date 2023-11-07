#!/bin/bash

PROJECT_DIR=$(pwd)

run_isort_on_file() {
    file="$1"
    if [ -f "$file" ] && [[ "$file" == *.py ]]; then
        isort "$file"
        echo "Ran isort on: $file"
    fi
}

run_isort_on_directory() {
    directory="$1"
    for file in "$directory"/*; do
        if [ -d "$file" ]; then
            run_isort_on_directory "$file"
        else
            run_isort_on_file "$file"
        fi
    done
}

run_isort_on_project() {
    project_dir="$1"
    run_isort_on_directory "$project_dir"
}

run_isort_on_project "$PROJECT_DIR"
