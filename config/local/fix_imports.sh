#!/bin/bash

# Get the current directory where the script is run
PROJECT_DIR=$(pwd)

# Function to run isort on a single file
run_isort_on_file() {
    file="$1"
    if [ -f "$file" ] && [[ "$file" == *.py ]]; then
        isort "$file"
        echo "Ran isort on: $file"
    fi
}

# Function to run isort on all Python files in a directory
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

# Main function to run isort on the current directory
run_isort_on_project() {
    project_dir="$1"
    run_isort_on_directory "$project_dir"
}

# Run isort on the current directory
run_isort_on_project "$PROJECT_DIR"
