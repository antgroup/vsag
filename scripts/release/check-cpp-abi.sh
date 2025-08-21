#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
TARGET_FILENAME="libvsag.so"
ARCHIVE_FILE="$1"

# --- Script Logic ---

# 1. Validate input
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_archive.tar.gz>"
    exit 1
fi

if [ ! -f "$ARCHIVE_FILE" ]; then
    echo "Error: Archive not found at '$ARCHIVE_FILE'"
    exit 1
fi

echo "🔍 Searching for '$TARGET_FILENAME' in '$ARCHIVE_FILE'..."

# 2. Find the full path of the file inside the archive
# We use grep to find the line ending in our target filename to be precise.
FILE_PATH_IN_ARCHIVE=$(tar --list --file="$ARCHIVE_FILE" | grep "/${TARGET_FILENAME}$")

if [ -z "$FILE_PATH_IN_ARCHIVE" ]; then
    echo "❌ Error: '$TARGET_FILENAME' not found within the archive."
    exit 1
fi

# In case there are multiple matches, we'll use the first one and warn the user.
if [ "$(echo "$FILE_PATH_IN_ARCHIVE" | wc -l)" -gt 1 ]; then
    echo "⚠️ Warning: Multiple instances of '$TARGET_FILENAME' found. Using the first one:"
    FILE_PATH_IN_ARCHIVE=$(echo "$FILE_PATH_IN_ARCHIVE" | head -n 1)
fi
echo "  ✔️ Found at: $FILE_PATH_IN_ARCHIVE"


# 3. Create a temporary directory for extraction
# The `trap` command ensures the temp directory is removed when the script exits,
# even if it fails.
TMP_DIR=$(mktemp -d)
trap 'echo "🧹 Cleaning up temporary files..."; rm -rf -- "$TMP_DIR"' EXIT

echo "📦 Extracting file to a temporary location..."
# Extract the specific file (-C changes to the temp directory first)
tar --extract --file="$ARCHIVE_FILE" --directory="$TMP_DIR" "$FILE_PATH_IN_ARCHIVE"

EXTRACTED_FILE_PATH="$TMP_DIR/$FILE_PATH_IN_ARCHIVE"

# 4. Check the ABI of the extracted file
echo "🔬 Analyzing C++ ABI for '$TARGET_FILENAME'..."

# The command to check the ABI. We look for the 'cxx11' marker in string-related symbols.
# `strings` outputs printable character sequences from the file.
# `grep -q` runs in quiet mode, returning a status code (0 for found, 1 for not found).
if strings "$EXTRACTED_FILE_PATH" | grep -q 'std::__cxx11::basic_string'; then
    echo "✅ SUCCESS: The library '$TARGET_FILENAME' was compiled with the CXX11 ABI."
    exit 0
else
    echo "❌ FAILURE: The library '$TARGET_FILENAME' was NOT compiled with the CXX11 ABI (or is stripped)."
    exit 1
fi
