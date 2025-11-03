#!/bin/bash

# Check if the file exists
if [ ! -f ".ci/auto_settings.sav" ]; then
    echo "Error: auto_settings.sav file not found"
    exit 1
fi

# Read the file line by line
while IFS= read -r line; do
    # Skip empty lines
    if [ -n "$line" ]; then
        # Execute caput command with the line contents as arguments
        caput $line
    fi
done < ".ci/auto_settings.sav"
