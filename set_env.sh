#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set PROJECT_ROOT to the script directory
export PROJECT_ROOT="$SCRIPT_DIR"
echo "Set PROJECT_ROOT to $PROJECT_ROOT"

# Execute the command passed as arguments
"$@"