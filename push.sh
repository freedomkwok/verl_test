#!/bin/bash
# Save current directory
CURRENT_DIR=$(pwd)

# Push verl submodule
git add . && git commit -m "Update" && git push

