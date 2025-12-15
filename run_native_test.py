#!/usr/bin/env python3
"""
Native Polysome Data Parallel Test Runner

This script runs the data parallel vLLM test directly on the host machine without Docker.
It can be configured to use either real data paths or mock/test paths.

Usage:
    # With real data paths (default)
    python run_native_test.py
    
    # With mock/test data (for quick testing)
    python run_native_test.py --mock-data
    
    # With custom GPU selection
    CUDA_VISIBLE_DEVICES=6,7 python run_native_test.py
    
    # With custom workflow
    python run_native_test.py --workflow workflows/my_workflow.json
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add the source directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def setup_environment(use_mock_data=False, custom_workflow=None):
    """Set up environment variables to mimic Docker environment."""
    
    current_dir = Path(__file__).parent
    
    if use_mock_data:
        # Use local paths for testing without real data
        model_path = str(current_dir / "test_models")  # Mock model path
        data_path = str(current_dir / "test_data")     # Mock data path
        output_path = str(current_dir / "test_output") # Mock output path
        print("Using mock data paths for testing")
    else:
        # Use real data paths from environment variables or defaults
        model_path = os.getenv("MODEL_PATH", "./models")
        data_path = os.getenv("DATA_PATH", "./data/input")
        output_path = os.getenv("OUTPUT_PATH", "./data/output")
        print("Using real data paths")
    
    prompts_path = current_dir / "prompts"
    workflows_path = current_dir / "workflows"
    
    # Determine workflow file
    if custom_workflow:
        workflow_file = custom_workflow
    else:
        workflow_file = str(workflows_path / "test_vllm_data_parallel_basic.json")
    
    # Set default paths if not already set
    env_vars = {
        "WORKFLOW_PATH": workflow_file,
        "VLLM_USE_V1": "1",  # Required for data parallelism
        "VLLM_LOGGING_LEVEL": "INFO",
        "MODEL_PATH": model_path,
        "DATA_PATH": data_path,
        "OUTPUT_PATH": output_path,
        "PROMPTS_PATH": str(prompts_path),
        "WORKFLOWS_PATH": str(workflows_path),
    }
    
    # Override with environment variables
    for key, default_value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = default_value
    
    # Set up CUDA devices if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
        print(f"Setting CUDA_VISIBLE_DEVICES=6,7 (default)")
    else:
        print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Print environment setup
    print("=== Native Test Environment Setup ===")
    print(f"WORKFLOW_PATH: {os.environ['WORKFLOW_PATH']}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"VLLM_USE_V1: {os.environ['VLLM_USE_V1']}")
    print(f"MODEL_PATH: {os.environ['MODEL_PATH']}")
    print(f"DATA_PATH: {os.environ['DATA_PATH']}")
    print(f"OUTPUT_PATH: {os.environ['OUTPUT_PATH']}")
    print("")


def validate_paths(use_mock_data=False):
    """Validate that required paths exist."""
    
    # Required paths
    paths_to_check = {
        "workflow_file": Path(os.environ["WORKFLOW_PATH"]),
        "src_dir": Path(__file__).parent / "src",
        "workflows_dir": Path(os.environ["WORKFLOWS_PATH"]),
        "prompts_dir": Path(os.environ["PROMPTS_PATH"]),
    }
    
    # Only check real paths if not using mock data
    if not use_mock_data:
        paths_to_check.update({
            "model_dir": Path(os.environ["MODEL_PATH"]),
            "data_dir": Path(os.environ["DATA_PATH"]),
            "input_file": Path(os.environ["DATA_PATH"]) / "metadata_without_results.json",
        })
    
    # Always check output directory (we'll create it if needed)
    paths_to_check["output_dir"] = Path(os.environ["OUTPUT_PATH"])
    
    print("=== Validating Required Paths ===")
    missing_paths = []
    
    for name, path in paths_to_check.items():
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} (MISSING)")
            missing_paths.append((name, path))
    
    if missing_paths:
        print(f"\nERROR: Missing required paths:")
        for name, path in missing_paths:
            print(f"  - {name}: {path}")
        
        if use_mock_data:
            print("\nWhen using --mock-data, some paths are expected to be missing.")
            print("The script will create mock data directories as needed.")
        else:
            print("\nPlease ensure all required paths exist before running the test.")
            return False
    
    print("✓ Path validation completed\n")
    return True


def create_directories():
    """Create output and mock directories if needed."""
    output_path = Path(os.environ["OUTPUT_PATH"])
    if not output_path.exists():
        print(f"Creating output directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Create the specific workflow output directory
    workflow_output = output_path / "test_vllm_data_parallel_basic"
    if not workflow_output.exists():
        print(f"Creating workflow output directory: {workflow_output}")
        workflow_output.mkdir(parents=True, exist_ok=True)


def create_mock_data():
    """Create mock data for testing purposes."""
    data_path = Path(os.environ["DATA_PATH"])
    model_path = Path(os.environ["MODEL_PATH"])
    
    # Create mock data directory
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Create mock input file
    mock_input_file = data_path / "metadata_without_results.json"
    if not mock_input_file.exists():
        print(f"Creating mock input file: {mock_input_file}")
        mock_data = [
            {"case_mapping": "test_case_1", "some_field": "test_data_1"},
            {"case_mapping": "test_case_2", "some_field": "test_data_2"},
        ]
        with open(mock_input_file, 'w') as f:
            json.dump(mock_data, f, indent=2)
    
    # Create mock model directory (just a placeholder)
    model_path.mkdir(parents=True, exist_ok=True)
    mock_model_dir = model_path / "gemma-3-27b-it-quantized.w4a16"
    mock_model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created mock data structure")


def patch_workflow_paths():
    """
    Patch the workflow to use the correct paths for native execution.
    This updates the workflow JSON to use the actual paths instead of Docker paths.
    """
    workflow_path = Path(os.environ["WORKFLOW_PATH"])
    
    # Read the workflow file
    with open(workflow_path, 'r') as f:
        workflow_data = json.load(f)
    
    # Update the paths to use environment variables
    workflow_data["data_dir"] = os.environ["DATA_PATH"]
    workflow_data["output_dir"] = os.environ["OUTPUT_PATH"]
    workflow_data["prompts_dir"] = os.environ["PROMPTS_PATH"]
    
    # Update model path in nodes if needed
    for node in workflow_data.get("nodes", []):
        if "params" in node and "model_name" in node["params"]:
            # Update model path to use the actual model directory
            model_name = node["params"]["model_name"]
            if model_name.startswith("/models/"):
                # Replace /models/ with the actual model path
                new_model_path = os.path.join(os.environ["MODEL_PATH"], model_name[8:])  # Remove /models/ prefix
                node["params"]["model_name"] = new_model_path
                print(f"Updated model path: {model_name} -> {new_model_path}")
    
    # Create a temporary workflow file with updated paths
    temp_workflow_path = workflow_path.parent / f"temp_{workflow_path.name}"
    with open(temp_workflow_path, 'w') as f:
        json.dump(workflow_data, f, indent=2)
    
    # Update the environment variable to use the temporary file
    os.environ["WORKFLOW_PATH"] = str(temp_workflow_path)
    print(f"Using temporary workflow file with updated paths: {temp_workflow_path}")


def main() -> int:
    """
    Main execution function for native testing.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Native Polysome Data Parallel Test Runner")
    parser.add_argument("--mock-data", action="store_true", 
                       help="Use mock data paths for testing (creates test data)")
    parser.add_argument("--workflow", type=str, 
                       help="Path to custom workflow JSON file")
    parser.add_argument("--gpus", type=str, default="6,7",
                       help="Comma-separated list of GPU IDs to use (default: 6,7)")
    
    args = parser.parse_args()
    
    # Set GPU environment if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    print("=== Polysome Data Parallel Native Test ===")
    print(f"Timestamp: {os.popen('date').read().strip()}")
    print(f"Mock data mode: {'ON' if args.mock_data else 'OFF'}")
    print("")
    
    # Set up environment
    setup_environment(use_mock_data=args.mock_data, custom_workflow=args.workflow)
    
    # Validate paths
    if not validate_paths(use_mock_data=args.mock_data):
        return 1
    
    # Create directories
    create_directories()
    
    # Create mock data if needed
    if args.mock_data:
        create_mock_data()
    
    # Patch workflow paths for native execution
    patch_workflow_paths()
    
    # Now import Polysome modules after environment setup
    try:
        from polysome.utils.logging import setup_logging
        from polysome.workflow import Workflow
    except ImportError as e:
        print(f"ERROR: Failed to import Polysome modules: {e}")
        print("Make sure you're running from the correct directory and all dependencies are installed.")
        return 1
    
    try:
        # Get workflow path from environment
        workflow_path = os.environ["WORKFLOW_PATH"]
        
        # Initialize workflow to get log directory and name
        workflow = Workflow(workflow_path)
        log_dir = workflow.get_log_dir()
        workflow_name = workflow.get_workflow_name()
        
        # Set up logging with workflow's log directory and name
        setup_logging(level=logging.INFO, log_dir=log_dir, workflow_name=workflow_name)
        logger = logging.getLogger(__name__)
        
        logger.info(f"Loading workflow from: {workflow_path}")
        logger.info(f"Workflow name: {workflow_name}")
        logger.info(f"Logs will be saved to: {log_dir}")
        success = workflow.run(validate_first=True)
        
        if success:
            logger.info("✓ Workflow completed successfully")
            print("\n=== Test Completed Successfully ===")
            return 0
        else:
            logger.error("✗ Workflow execution failed")
            print("\n=== Test Failed ===")
            return 1
            
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        print(f"\n=== Test Failed with Error ===")
        print(f"Error: {e}")
        return 1
    
    finally:
        # Clean up temporary workflow file
        temp_workflow_path = Path(os.environ["WORKFLOW_PATH"])
        if temp_workflow_path.name.startswith("temp_"):
            try:
                temp_workflow_path.unlink()
                print(f"Cleaned up temporary workflow file: {temp_workflow_path}")
            except:
                pass


if __name__ == "__main__":
    sys.exit(main())