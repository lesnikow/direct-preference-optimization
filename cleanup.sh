#!/bin/bash

# Check if the directory parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory> [--dry-run]"
  exit 1
fi

# Assign the directory parameter and check for --dry-run option
target_dir="$1"
dry_run=false
if [ "$2" == "--dry-run" ]; then
  dry_run=true
fi

# Move to the target directory
cd "$target_dir" || { echo "Directory $target_dir not found"; exit 1; }

# Get the list of step-* directories sorted by modification time in descending order
dirs=($(ls -dt step-* 2>/dev/null))

# Check if there are any step-* directories
if [ ${#dirs[@]} -gt 0 ]; then
  # Keep the most recent step-* directory
  recent_dir="${dirs[0]}"
  
  # Loop through all step-* directories
  for dir in "${dirs[@]}"; do
    # Remove the directory if it's not the most recent one
    if [ "$dir" != "$recent_dir" ]; then
      if $dry_run; then
        echo "Dry run: would remove $dir"
      else
        rm -rf "$dir"
        echo "Removed $dir"
      fi
    fi
  done
  
  echo "Only keeping the most recent step-* directory ($recent_dir) and LATEST directory."
else
  echo "No step-* directories found."
fi

