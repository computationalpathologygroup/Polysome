#!/bin/bash
# Integration Test Suite - Docker Method
# Tests Polysome workflows inside the Docker container

# set -e  # Don't exit on error so we see all results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TEST_ROOT/../.." && pwd)"
IMAGE_NAME="polysome-integration-test"

echo "=== Starting Polysome Docker Integration Tests ==="

# 1. Build the Docker image
echo "Building Docker image: $IMAGE_NAME..."
if ! docker build -t "$IMAGE_NAME" "$PROJECT_ROOT"; then
    echo "❌ Docker build failed."
    exit 1
fi

# 2. Create output directory if it doesn't exist
OUTPUT_DIR="$PROJECT_ROOT/output/integration_tests"
mkdir -p "$OUTPUT_DIR"

# 3. Define helper for running workflows in Docker
run_docker_workflow() {
    local workflow_name=$1
    local workflow_file=$2
    
    echo "--- Running workflow: $workflow_name ---"
    
    # We mount the relevant parts of the project into /opt/algorithm
    # to maintain the relative path structure used in the workflows.
    # We also mount the host's HF cache to avoid gated access issues and re-downloads.
    docker run --rm --gpus all \
        -v "$PROJECT_ROOT/tests/integration/data:/opt/algorithm/tests/integration/data" \
        -v "$PROJECT_ROOT/tests/integration/workflows:/opt/algorithm/tests/integration/workflows" \
        -v "$PROJECT_ROOT/src/polysome/templates/prompts:/opt/algorithm/src/polysome/templates/prompts" \
        -v "$OUTPUT_DIR:/opt/algorithm/output/integration_tests" \
        -v "$PROJECT_ROOT/models:/opt/algorithm/models" \
        -v "$PROJECT_ROOT/models:/models" \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        -e WORKFLOW_PATH="/opt/algorithm/tests/integration/workflows/$workflow_file" \
        "$IMAGE_NAME"
        
    if [ $? -eq 0 ]; then
        echo "✅ Workflow $workflow_name completed successfully."
        return 0
    else
        echo "❌ Workflow $workflow_name failed."
        return 1
    fi
}

# 4. Run workflows
FAILED_COUNT=0

# HuggingFace Test
run_docker_workflow "HuggingFace" "test_huggingface.json" || FAILED_COUNT=$((FAILED_COUNT+1))

# vLLM Test
run_docker_workflow "vLLM" "test_vllm.json" || FAILED_COUNT=$((FAILED_COUNT+1))

# llama_cpp Test
run_docker_workflow "llama_cpp" "test_llamacpp.json" || FAILED_COUNT=$((FAILED_COUNT+1))

echo "=== Docker Integration Tests Completed ($FAILED_COUNT failures) ==="
if [ $FAILED_COUNT -eq 0 ]; then
    exit 0
else
    exit 1
fi
