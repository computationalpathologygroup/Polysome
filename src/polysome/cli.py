"""
Polysome CLI - Command-line interface for Polysome framework.
"""

import argparse
import sys
import os
import shutil
import logging
from pathlib import Path
from typing import Optional

# Import core modules
from polysome.workflow import Workflow
from polysome.utils.logging import setup_logging


def get_templates_dir() -> Path:
    """Get the path to the templates directory in the package."""
    return Path(__file__).parent / "templates"


def init_project(project_name: Optional[str] = None, target_dir: Optional[str] = None) -> int:
    """
    Initialize a new Polysome project with example workflows and prompts.

    Args:
        project_name: Name of the project (default: "my-polysome-project")
        target_dir: Target directory (default: current directory)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    if project_name is None:
        project_name = "my-polysome-project"

    if target_dir is None:
        target_dir = os.getcwd()

    project_path = Path(target_dir) / project_name

    # Check if directory already exists
    if project_path.exists():
        print(f"Error: Directory '{project_path}' already exists")
        return 1

    print(f"Initializing Polysome project: {project_name}")
    print(f"Location: {project_path}")
    print()

    try:
        # Create project structure
        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / "data").mkdir(exist_ok=True)
        (project_path / "output").mkdir(exist_ok=True)
        (project_path / "workflows").mkdir(exist_ok=True)
        (project_path / "prompts").mkdir(exist_ok=True)

        templates_dir = get_templates_dir()

        # Copy example workflows
        workflows_src = templates_dir / "workflows"
        if workflows_src.exists():
            for item in workflows_src.iterdir():
                if item.is_file():
                    shutil.copy2(item, project_path / "workflows" / item.name)
                    print(f"  ✓ Created workflow: workflows/{item.name}")

        # Copy example prompts
        prompts_src = templates_dir / "prompts"
        if prompts_src.exists():
            for prompt_dir in prompts_src.iterdir():
                if prompt_dir.is_dir() and not prompt_dir.name.startswith("_"):
                    dest_prompt_dir = project_path / "prompts" / prompt_dir.name
                    shutil.copytree(prompt_dir, dest_prompt_dir)
                    print(f"  ✓ Created prompt set: prompts/{prompt_dir.name}/")

        # Create example input data
        example_data = project_path / "data" / "input.json"
        example_data.write_text('''[
  {"id": "1", "question": "What is machine learning?"},
  {"id": "2", "question": "Explain the concept of neural networks."},
  {"id": "3", "question": "What are the main types of AI?"}
]
''')
        print(f"  ✓ Created example data: data/input.json")

        # Create README
        readme_content = f'''# {project_name}

A Polysome project for LLM-based data generation.

## Getting Started

### 1. Review the Workflow

Edit `workflows/basic_text_generation.json` to customize the workflow:
- Modify the model name
- Adjust generation parameters
- Add or remove processing nodes

### 2. Customize Prompts

Edit prompts in `prompts/simple_qa/`:
- `system_prompt.txt`: Define the AI's role and behavior
- `user_prompt.txt`: Template for formatting inputs
- `few_shot.jsonl`: Example inputs and outputs

### 3. Prepare Your Data

Edit `data/input.json` with your own data. Ensure it has the fields referenced in your prompts.

### 4. Run the Workflow

```bash
polysome run workflows/basic_text_generation.json
```

Or use the Python API:

```python
from polysome.workflow import Workflow

workflow = Workflow("workflows/basic_text_generation.json")
workflow.run()
```

### 5. Check Results

Results will be saved to `output/` directory.

## Environment Variables

Configure paths using environment variables:
- `MODEL_PATH`: Path to model files (default: ./models)
- `DATA_PATH`: Path to data files (default: ./data)
- `OUTPUT_PATH`: Path for output files (default: ./output)

## Learn More

- [Polysome Documentation](https://github.com/computationalpathologygroup/Polysome)
- [Full Documentation](https://github.com/computationalpathologygroup/Polysome/tree/main/docs)
'''
        (project_path / "README.md").write_text(readme_content)
        print(f"  ✓ Created: README.md")

        # Create .gitignore
        gitignore_content = '''# Polysome outputs
output/
*.log

# Models (usually too large for git)
models/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
'''
        (project_path / ".gitignore").write_text(gitignore_content)
        print(f"  ✓ Created: .gitignore")

        print()
        print(f"✓ Project initialized successfully!")
        print()
        print("Next steps:")
        print(f"  cd {project_name}")
        print("  # Edit workflows/basic_text_generation.json and prompts/")
        print("  polysome run workflows/basic_text_generation.json")
        print()

        return 0

    except Exception as e:
        print(f"Error initializing project: {e}")
        return 1


def run_workflow(workflow_path: str, validate_first: bool = True, log_level: str = "INFO") -> int:
    """
    Run a Polysome workflow.

    Args:
        workflow_path: Path to the workflow JSON file
        validate_first: Whether to validate before running (default: True)
        log_level: Logging level (default: INFO)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Load workflow
        workflow = Workflow(workflow_path)

        # Setup logging
        log_dir = workflow.get_log_dir()
        workflow_name = workflow.get_workflow_name()

        level = getattr(logging, log_level.upper(), logging.INFO)
        setup_logging(level=level, log_dir=log_dir, workflow_name=workflow_name)

        logger = logging.getLogger(__name__)
        logger.info(f"Starting workflow: {workflow_name}")
        logger.info(f"Workflow file: {workflow_path}")

        # Run workflow
        success = workflow.run(validate_first=validate_first)

        if success:
            logger.info("Workflow completed successfully")
            print(f"\n✓ Workflow completed successfully!")
            print(f"Results saved to: {workflow.output_dir}")
            return 0
        else:
            logger.error("Workflow execution failed")
            print(f"\n✗ Workflow execution failed")
            print(f"Check logs in: {log_dir}")
            return 1

    except Exception as e:
        print(f"Error running workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Polysome - Domain-agnostic LLM data generation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize a new project
  polysome init my-project

  # Run a workflow
  polysome run workflows/my_workflow.json

  # Run with debug logging
  polysome run workflows/my_workflow.json --log-level DEBUG

For more information: https://github.com/computationalpathologygroup/Polysome
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new Polysome project with example workflows and prompts"
    )
    init_parser.add_argument(
        "project_name",
        nargs="?",
        default="my-polysome-project",
        help="Name of the project directory to create (default: my-polysome-project)"
    )
    init_parser.add_argument(
        "--dir",
        dest="target_dir",
        help="Target directory (default: current directory)"
    )

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a Polysome workflow"
    )
    run_parser.add_argument(
        "workflow_path",
        help="Path to the workflow JSON file"
    )
    run_parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        default=True,
        help="Skip workflow validation before running"
    )
    run_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    # Version command
    parser.add_argument(
        "--version",
        action="version",
        version=f"Polysome {get_version()}"
    )

    args = parser.parse_args()

    # Execute command
    if args.command == "init":
        return init_project(args.project_name, args.target_dir)
    elif args.command == "run":
        return run_workflow(args.workflow_path, args.validate, args.log_level)
    else:
        parser.print_help()
        return 1


def get_version() -> str:
    """Get the package version."""
    try:
        from polysome import __version__
        return __version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
