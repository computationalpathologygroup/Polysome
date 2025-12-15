#!/bin/bash

# Run Polysome container with histai_combined_vllm workflow
# This script uses the new configurable test script with custom mount paths

set -e

# Set CUDA devices to use GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

echo "=== Running Polysome with HistAI vLLM Workflow ==="
echo "Using CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Timestamp: $(date)"
echo ""

# Configuration
MODEL_PATH="${MODEL_PATH:-./models}"
DATA_PATH="${DATA_PATH:-./data/input}"
OUTPUT_PATH="${OUTPUT_PATH:-./data/output}"
PROMPTS_PATH="$(pwd)/prompts"
WORKFLOWS_PATH="$(pwd)/workflows"
WORKFLOW_NAME="regenerate_questions_workflow"
# Verify paths exist
echo "Checking required paths..."
required_paths=(
  "$MODEL_PATH"
  "$DATA_PATH"
  "$OUTPUT_PATH"
  "$PROMPTS_PATH"
  "$WORKFLOWS_PATH/$WORKFLOW_NAME.json"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: Required path missing: $path"
    exit 1
  fi
  echo "✓ Found: $path"
done

# Check for input file
INPUT_FILE="$DATA_PATH/metadata_without_results.json"
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "ERROR: Input file not found: $INPUT_FILE"
  exit 1
fi
echo "✓ Found input file: $INPUT_FILE"

echo ""
echo "Running container with the following configuration:"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_PATH"
echo "  Output: $OUTPUT_PATH"
echo "  Prompts: $PROMPTS_PATH"
echo "  Workflows: $WORKFLOWS_PATH"
echo "  Workflow: $WORKFLOW_NAME.json"
echo "  Input file: metadata_without_results.json"
echo ""

# Run the configurable test script
./test_container_configurable.sh \
  -t gpu \
  -w "$WORKFLOW_NAME" \
  -m "$MODEL_PATH" \
  -d "$DATA_PATH" \
  -o "$OUTPUT_PATH" \
  --prompts-path "$PROMPTS_PATH" \
  --workflows-path "$WORKFLOWS_PATH"

echo ""
echo "=== HistAI vLLM Workflow Complete ==="
