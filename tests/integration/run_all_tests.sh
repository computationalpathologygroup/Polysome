#!/bin/bash
# Master Integration Test Runner for Polysome

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "===================================================="
echo "   Polysome Multi-Engine Integration Test Suite     "
echo "===================================================="

# Ensure we are in the project root
cd "$PROJECT_ROOT"

# Make scripts executable
chmod +x tests/integration/scripts/*.sh
chmod +x tests/integration/scripts/*.py

# 1. CLI Tests
echo ""
echo "[Step 1/4] Running CLI Integration Tests..."
bash tests/integration/scripts/test_cli.sh

# 2. API Tests
echo ""
echo "[Step 2/4] Running API Integration Tests..."
python3 tests/integration/scripts/test_api.py

# 3. Docker Tests (Optional, requires Docker)
if command -v docker &> /dev/null; then
    echo ""
    echo "[Step 3/4] Running Docker Integration Tests..."
    bash tests/integration/scripts/test_docker.sh
else
    echo ""
    echo "[Step 3/4] Skipping Docker tests (Docker not found)."
fi

# 4. Final Validation
echo ""
echo "[Step 4/4] Validating All Outputs..."
python3 tests/integration/scripts/validate_outputs.py

echo ""
echo "===================================================="
echo "         Integration Tests Completed!               "
echo "===================================================="
