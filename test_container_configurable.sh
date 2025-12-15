#!/bin/bash

# Configurable test script for Polysome containers
# Supports both GPU and ARM64 CPU containers with configurable mount points

set -e # Exit on any error

# Default configuration
CONTAINER_TYPE="arm64" # Default to arm64
WORKFLOW_NAME=""
DOCKER_ARGS=""

# Default mount locations (can be overridden)
MODEL_PATH="$(pwd)/test/model"
DATA_PATH="$(pwd)/test/data"
OUTPUT_PATH="$(pwd)/test/output"
WORKFLOWS_PATH="$(pwd)/test/workflows"
PROMPTS_PATH="$(pwd)/prompts"

# Function to show usage
show_usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -t, --type TYPE           Container type: 'gpu' or 'arm64' (default: arm64)"
  echo "  -w, --workflow NAME       Workflow name (default: auto-detected based on type)"
  echo "  -m, --model-path PATH     Path to model directory (default: ./test/model)"
  echo "  -d, --data-path PATH      Path to data directory (default: ./test/data)"
  echo "  -o, --output-path PATH    Path to output directory (default: ./test/output)"
  echo "  --workflows-path PATH     Path to workflows directory (default: ./test/workflows)"
  echo "  --prompts-path PATH       Path to prompts directory (default: ./prompts)"
  echo "  -h, --help               Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0                        # Test ARM64 container with default paths"
  echo "  $0 -t gpu                 # Test GPU container with default paths"
  echo "  $0 -m /path/to/models/gemma-3-27b-it-quantized.w4a16 \\"
  echo "     -d /path/to/data/input \\"
  echo "     -o /path/to/data/output \\"
  echo "     --prompts-path /path/to/prompts"
  echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
  -t | --type)
    CONTAINER_TYPE="$2"
    shift 2
    ;;
  -w | --workflow)
    WORKFLOW_NAME="$2"
    shift 2
    ;;
  -m | --model-path)
    MODEL_PATH="$2"
    shift 2
    ;;
  -d | --data-path)
    DATA_PATH="$2"
    shift 2
    ;;
  -o | --output-path)
    OUTPUT_PATH="$2"
    shift 2
    ;;
  --workflows-path)
    WORKFLOWS_PATH="$2"
    shift 2
    ;;
  --prompts-path)
    PROMPTS_PATH="$2"
    shift 2
    ;;
  -h | --help)
    show_usage
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    show_usage
    exit 1
    ;;
  esac
done

# Validate container type
if [[ "$CONTAINER_TYPE" != "gpu" && "$CONTAINER_TYPE" != "arm64" ]]; then
  echo "ERROR: Container type must be 'gpu' or 'arm64'"
  show_usage
  exit 1
fi

# Set container-specific configuration
if [[ "$CONTAINER_TYPE" == "gpu" ]]; then
  CONTAINER_NAME="polysome-runner"
  DOCKERFILE="Dockerfile"
  DOCKER_ARGS="--gpus all"
  DEFAULT_WORKFLOW="gpu_test"
  CONTAINER_DESC="GPU-enabled"
else
  CONTAINER_NAME="polysome-runner-arm64"
  DOCKERFILE="Dockerfile.arm64"
  DOCKER_ARGS=""
  DEFAULT_WORKFLOW="arm64_test"
  CONTAINER_DESC="ARM64 CPU-only"
fi

# Set workflow name
if [[ -z "$WORKFLOW_NAME" ]]; then
  WORKFLOW_NAME="$DEFAULT_WORKFLOW"
fi

# Configuration
LOG_FILE="$(pwd)/test_${CONTAINER_TYPE}_output.log"

echo "=== Polysome ${CONTAINER_DESC} Container Test ==="
echo "Timestamp: $(date)"
echo "Container: ${CONTAINER_NAME}"
echo "Workflow: ${WORKFLOW_NAME}.json"
echo ""

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_PATH}"

# Check if container image exists
echo "1. Checking if ${CONTAINER_DESC} container image exists..."
if ! docker images | grep -q "${CONTAINER_NAME}"; then
  echo "   ERROR: Container image '${CONTAINER_NAME}' not found!"
  echo "   Please build it first with: docker build -f ${DOCKERFILE} -t ${CONTAINER_NAME} ."
  exit 1
fi
echo "   ✓ Container image found"

# Check if required directories and files exist
echo ""
echo "2. Checking test setup..."
required_paths=(
  "${DATA_PATH}"
  "${WORKFLOWS_PATH}/${WORKFLOW_NAME}.json"
  "${MODEL_PATH}"
  "${PROMPTS_PATH}"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "   ERROR: Required path missing: $path"
    if [[ "$path" == *"${WORKFLOW_NAME}.json" ]]; then
      echo "   Available workflows:"
      ls -1 "${WORKFLOWS_PATH}/"*.json 2>/dev/null | sed 's/.*\//     /' || echo "     No workflows found"
    fi
    exit 1
  fi
  if [[ -d "$path" ]]; then
    echo "   ✓ Found directory: $(basename "$path")"
  else
    echo "   ✓ Found file: $(basename "$path")"
  fi
done

# Show test configuration
echo ""
echo "3. Test configuration:"
echo "   Container type: ${CONTAINER_DESC}"
echo "   Container name: ${CONTAINER_NAME}"
echo "   Workflow: ${WORKFLOW_NAME}.json"
echo "   Model path: ${MODEL_PATH}"
echo "   Data path: ${DATA_PATH}"
echo "   Output path: ${OUTPUT_PATH}"
echo "   Workflows path: ${WORKFLOWS_PATH}"
echo "   Prompts path: ${PROMPTS_PATH}"
echo "   Log file: ${LOG_FILE}"
if [[ -n "$DOCKER_ARGS" ]]; then
  echo "   Docker args: ${DOCKER_ARGS}"
fi

# Build docker command
DOCKER_CMD="docker run --rm"
if [[ -n "$DOCKER_ARGS" ]]; then
  DOCKER_CMD="$DOCKER_CMD $DOCKER_ARGS"
fi
DOCKER_CMD="$DOCKER_CMD \
  --user $(id -u):$(id -g) \
  -v ${MODEL_PATH}:/models \
  -v ${DATA_PATH}:/data \
  -v ${OUTPUT_PATH}:/output \
  -v ${WORKFLOWS_PATH}:/workflows \
  -v ${PROMPTS_PATH}:/prompts \
  -v ${VLLM_CACHE_PATH:-./vllm_cache}:/root/.cache \
  -e WORKFLOW_PATH=/workflows/${WORKFLOW_NAME}.json \
  -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  ${CONTAINER_NAME}"

# Run the container
echo ""
echo "5. Running ${CONTAINER_DESC} container..."
echo "   Command: ${DOCKER_CMD}"
echo ""
echo "   Starting container (this may take several minutes for model loading and inference)..."
echo "   Logs will be saved to: ${LOG_FILE}"

# Run container and capture all output
eval "$DOCKER_CMD" 2>&1 | tee "${LOG_FILE}"

# Capture the exit code from docker run
DOCKER_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "6. Analyzing results..."

# Check exit code
if [[ $DOCKER_EXIT_CODE -eq 0 ]]; then
  echo "   ✓ Container completed successfully (exit code: 0)"
else
  echo "   ✗ Container failed (exit code: $DOCKER_EXIT_CODE)"
fi

# Check output files
echo ""
echo "7. Output files generated:"
if [[ -d "${OUTPUT_PATH}" ]]; then
  output_files=$(find "${OUTPUT_PATH}" -name "*.jsonl" -o -name "*.json" 2>/dev/null || true)
  if [[ -n "$output_files" ]]; then
    echo "$output_files" | while IFS= read -r file; do
      if [[ -f "$file" ]]; then
        size=$(wc -l <"$file" 2>/dev/null || echo "0")
        echo "   ✓ $(basename "$file") ($size lines)"

        # Show first few lines of output for verification
        if [[ "$size" -gt 0 && $(basename "$file") == *.jsonl ]]; then
          echo "     Preview: $(head -1 "$file" 2>/dev/null | cut -c1-80)..."
        fi
      fi
    done
  else
    echo "   ✗ No output files found"
  fi
else
  echo "   ✗ Output directory not found"
fi

# Show log summary
echo ""
echo "8. Log summary:"
if [[ -f "${LOG_FILE}" ]]; then
  log_lines=$(wc -l <"${LOG_FILE}")
  echo "   Log file size: $log_lines lines"

  # Show performance info if available
  if grep -q "Model loaded" "${LOG_FILE}" 2>/dev/null; then
    echo "   ✓ Model loaded successfully"
  fi
  if grep -q "Workflow completed" "${LOG_FILE}" 2>/dev/null; then
    echo "   ✓ Workflow completed"
  fi

  # Count errors and warnings (look for actual log levels, not just words)
  error_count=$(grep -ic " - ERROR - \| - CRITICAL - " "${LOG_FILE}" 2>/dev/null || echo "0")
  warning_count=$(grep -ic " - WARNING - " "${LOG_FILE}" 2>/dev/null || echo "0")

  if [[ $error_count -gt 0 ]]; then
    echo "   ⚠ Errors found: $error_count"
  fi
  if [[ $warning_count -gt 0 ]]; then
    echo "   ⚠ Warnings found: $warning_count"
  fi

  echo ""
  echo "   To view full logs: cat ${LOG_FILE}"
  echo "   To view errors only: grep -i error ${LOG_FILE}"
  echo "   To view warnings: grep -i warning ${LOG_FILE}"
else
  echo "   ✗ Log file not created"
fi

# Final status
echo ""
echo "=== Test Summary ==="
if [[ $DOCKER_EXIT_CODE -eq 0 ]]; then
  echo "Status: SUCCESS ✓"
  echo "The ${CONTAINER_DESC} container completed the test workflow successfully."

  # Performance note for ARM64
  if [[ "$CONTAINER_TYPE" == "arm64" ]]; then
    echo ""
    echo "Note: ARM64 CPU inference is expected to be slower than GPU inference."
    echo "For production workloads, consider using the GPU container or smaller models."
  fi
else
  echo "Status: FAILED ✗"
  echo "The ${CONTAINER_DESC} container encountered an error. Check the logs above."
fi

echo ""
echo "Next steps:"
echo "  - Review logs: cat ${LOG_FILE}"
echo "  - Check output: ls -la ${OUTPUT_PATH}/"
echo "  - Examine workflow results in the output JSONL files"
if [[ "$CONTAINER_TYPE" == "arm64" ]]; then
  echo "  - Compare with GPU performance: $0 -t gpu [same mount options]"
else
  echo "  - Compare with ARM64 performance: $0 -t arm64 [same mount options]"
fi

exit $DOCKER_EXIT_CODE

