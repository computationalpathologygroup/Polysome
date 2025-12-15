#!/usr/bin/env python3
"""
Generic Polysome Workflow Runner

This script serves as a minimal entrypoint for running Polysome workflows in containerized environments.
It reads a workflow configuration from the path specified in the WORKFLOW_PATH environment variable
and executes it using the Polysome framework.

Environment Variables:
    WORKFLOW_PATH: Path to the workflow JSON file (default: /workflows/default.json)

Usage:
    python inference.py
    
    Or with custom workflow path:
    WORKFLOW_PATH=/workflows/my_workflow.json python inference.py
"""

import os
import sys
import logging
from pathlib import Path

# Add the source directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from polysome.utils.logging import setup_logging
from polysome.workflow import Workflow


def main() -> int:
    """
    Main execution function for workflow processing.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Get workflow path from environment
        workflow_path = os.getenv("WORKFLOW_PATH", "/workflows/default.json")
        
        # Validate workflow file exists
        if not Path(workflow_path).exists():
            # Set up basic logging for error reporting
            setup_logging(level=logging.INFO)
            logger = logging.getLogger(__name__)
            logger.error(f"Workflow file not found: {workflow_path}")
            return 1
        
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
            logger.info("Workflow completed successfully")
            return 0
        else:
            logger.error("Workflow execution failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())