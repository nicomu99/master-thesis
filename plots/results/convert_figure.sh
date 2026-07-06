#!/usr/bin/env bash

set -euo pipefail

INPUT_DIR="${1:-.}"
OUTPUT_DIR="${2:-pdfa}"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.pdf; do
    # Skip if no PDFs are found
    [ -e "$file" ] || continue

    filename="$(basename "$file")"
    output_file="$OUTPUT_DIR/$filename"

    echo "Converting: $file -> $output_file"

    gs \
      -dPDFA=1 \
      -dBATCH \
      -dNOPAUSE \
      -dNOOUTERSAVE \
      -sDEVICE=pdfwrite \
      -sColorConversionStrategy=RGB \
      -sProcessColorModel=DeviceRGB \
      -sOutputICCProfile=/usr/share/ghostscript/iccprofiles/srgb.icc \
      -dPDFACompatibilityPolicy=1 \
      -sOutputFile="$output_file" \
      "$file"
done

echo "Done."
