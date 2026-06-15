#!/bin/bash

# Default paths if not provided
VANILLA_JSON=${1:-"/data1/local/lerobot-VAI/outputs/eval/2026-01-21/22-24-16_smolvla_10_vanilla/eval_info.json"}
BASIS_JSON=${2:-"/data1/local/lerobot-VAI/outputs/eval/2026-01-22/13-05-07_smolvla_10_basis_concat_95000/eval_info.json"}
OUTPUT_CSV=${3:-"libero_object_merged_eval_results.csv"}

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Streamlining Evaluation Analysis..."
echo "Vanilla JSON: $VANILLA_JSON"
echo "Basis JSON:   $BASIS_JSON"
echo "Output CSV:   $OUTPUT_CSV"
echo ""

# Run single analysis for each to ensure individual CSVs are also generated if needed
# (Optional, but helps keep tracks)
python3 $SCRIPTS_DIR/analyze_eval.py "$VANILLA_JSON" --csv "vanilla_results.csv"
python3 $SCRIPTS_DIR/analyze_eval.py "$BASIS_JSON" --csv "basis_results.csv"

echo ""
echo "Merging results..."
python3 $SCRIPTS_DIR/merge_eval_results.py --vanilla "$VANILLA_JSON" --basis "$BASIS_JSON" --output "$OUTPUT_CSV"

echo ""
echo "Process complete. Results available in:"
echo " - vanilla_results.csv"
echo " - basis_results.csv"
echo " - $OUTPUT_CSV"
