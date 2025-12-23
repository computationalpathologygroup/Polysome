#!/usr/bin/env python3
"""
Integration Test Suite - API Method
Tests all three inference engines via programmatic API
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from polysome.workflow import Workflow
from polysome import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = TEST_ROOT / "workflows"
TESTS = [
    {
        "name": "HuggingFace Engine",
        "workflow": WORKFLOWS_DIR / "test_huggingface.json",
        "engine": "huggingface"
    },
    {
        "name": "vLLM Engine",
        "workflow": WORKFLOWS_DIR / "test_vllm.json",
        "engine": "vllm"
    },
    {
        "name": "llama.cpp Engine",
        "workflow": WORKFLOWS_DIR / "test_llamacpp.json",
        "engine": "llama_cpp"
    }
]

def run_single_test(test_config: dict) -> bool:
    """
    Run a single workflow test via API.

    Args:
        test_config: Dictionary with test configuration

    Returns:
        True if test passed, False otherwise
    """
    logger.info("=" * 60)
    logger.info(f"Testing: {test_config['name']}")
    logger.info("=" * 60)

    workflow_path = test_config['workflow']

    if not workflow_path.exists():
        logger.error(f"Workflow file not found: {workflow_path}")
        return False

    try:
        # Initialize workflow
        logger.info(f"Loading workflow from: {workflow_path}")
        workflow = Workflow(str(workflow_path))

        # Log workflow info
        logger.info(f"Workflow name: {workflow.get_workflow_name()}")
        logger.info(f"Output directory: {workflow.output_dir}")
        logger.info(f"Log directory: {workflow.get_log_dir()}")

        # Run workflow with validation
        logger.info("Starting workflow execution...")
        success = workflow.run(validate_first=True)

        if success:
            logger.info(f"✓ {test_config['name']} test PASSED")
            return True
        else:
            logger.error(f"✗ {test_config['name']} test FAILED")
            return False

    except Exception as e:
        logger.exception(f"Error during {test_config['name']}: {e}")
        return False

def main():
    """Run all API integration tests."""
    logger.info("=" * 60)
    logger.info("Polysome Integration Tests - API Method")
    logger.info(f"Version: {__version__}")
    logger.info("=" * 60)
    logger.info("")

    results = []

    for test in TESTS:
        passed = run_single_test(test)
        results.append({
            "name": test["name"],
            "passed": passed
        })
        logger.info("")

    # Validate outputs
    logger.info("=" * 60)
    logger.info("Validating Outputs")
    logger.info("=" * 60)

    validation_script = Path(__file__).parent / "validate_outputs.py"
    if validation_script.exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(validation_script)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("✓ Output validation passed")
            validation_passed = True
        else:
            logger.error("✗ Output validation failed")
            logger.error(result.stderr)
            validation_passed = False
    else:
        logger.warning("Validation script not found, skipping")
        validation_passed = True

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed_count = sum(1 for r in results if r["passed"])
    failed_count = len(results) - passed_count

    for result in results:
        status = "✓ PASSED" if result["passed"] else "✗ FAILED"
        logger.info(f"{result['name']}: {status}")

    logger.info(f"\nPassed: {passed_count}/{len(results)}")
    logger.info(f"Failed: {failed_count}/{len(results)}")

    if failed_count == 0 and validation_passed:
        logger.info("\n✓ All tests passed!")
        return 0
    else:
        logger.error("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
