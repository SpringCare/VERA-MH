#!/usr/bin/env bash
# Iterate over directories in to_be_truncated, truncate each to 30 turns,
# and write to truncated/ with dir name updated from t100 to t30.

set -e

INPUT_BASE="/Users/josh.gieringer/Desktop/HEOR AIM 3/to_be_truncated"
OUTPUT_BASE="/Users/josh.gieringer/Desktop/HEOR AIM 3/truncated"
TURNS=30
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_BASE"

for dir in "$INPUT_BASE"/*/; do
  [[ -d "$dir" ]] || continue
  name="$(basename "$dir")"
  out_name="${name//t100/t30}"
  in_path="$INPUT_BASE/$name"
  out_path="$OUTPUT_BASE/$out_name"
  echo "Truncating: $name -> $out_name ($TURNS turns)"
  python3 "$SCRIPT_DIR/truncate_conversations.py" -i "$in_path" -o "$out_path" -t "$TURNS"
done

echo "Done. Output in $OUTPUT_BASE"
