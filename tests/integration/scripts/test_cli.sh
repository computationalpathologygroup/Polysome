#!/bin/bash
# Integration Test Suite - CLI Method
# Tests all three inference engines via CLI

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TEST_ROOT/../.." && pwd)"

echo "======================================"
echo "Polysome Integration Tests - CLI"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "$PROJECT_ROOT/output/integration_tests"

# Function to run a single test
run_test() {
    local engine=$1
    local workflow=$2

    echo -e "${YELLOW}Testing $engine engine...${NC}"

    # Run the workflow
    if polysome run "$workflow" --log-level INFO; then
        echo -e "${GREEN}✓ $engine test completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ $engine test failed${NC}"
        return 1
    fi
}

# Test counter
PASSED=0
FAILED=0

# Test 1: HuggingFace
echo ""
echo "========================================"
echo "Test 1/3: HuggingFace Engine"
echo "========================================"
if run_test "HuggingFace" "$TEST_ROOT/workflows/test_huggingface.json"; then
    PASSED=$((PASSED+1))
else
    FAILED=$((FAILED+1))
fi

# Test 2: vLLM
echo ""
echo "========================================"
echo "Test 2/3: vLLM Engine"
echo "========================================"
if run_test "vLLM" "$TEST_ROOT/workflows/test_vllm.json"; then
    PASSED=$((PASSED+1))
else
    FAILED=$((FAILED+1))
fi

# Test 3: llama_cpp
echo ""
echo "========================================"
echo "Test 3/3: llama.cpp Engine (GGUF)"
echo "========================================"
echo -e "${YELLOW}Note: Requires GGUF model at /models/gemma-3-4b-it-qat-q4_0.gguf${NC}"

if run_test "llama.cpp" "$TEST_ROOT/workflows/test_llamacpp.json"; then
    PASSED=$((PASSED+1))
else
    FAILED=$((FAILED+1))
    echo -e "${YELLOW}If model not found, download from: https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf${NC}"
fi

# Validate outputs
echo ""
echo "========================================"
echo "Validating Outputs"
echo "========================================"
if python "$SCRIPT_DIR/validate_outputs.py"; then
    echo -e "${GREEN}✓ Output validation passed${NC}"
else
    echo -e "${RED}✗ Output validation failed${NC}"
    FAILED=$((FAILED+1))
fi

# Summary
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
