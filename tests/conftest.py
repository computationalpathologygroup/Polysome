import pytest
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock


@pytest.fixture
def temp_workspace():
    """
    Creates a temporary workspace with data and output directories.
    Automatically cleaned up after test.
    """
    temp_dir = Path(tempfile.mkdtemp())
    workspace = {
        "root": temp_dir,
        "data_dir": temp_dir / "data",
        "output_dir": temp_dir / "output",
    }

    # Create directories
    workspace["data_dir"].mkdir(parents=True)
    workspace["output_dir"].mkdir(parents=True)

    yield workspace

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_text_data():
    """Sample data for text processing tests."""
    return [
        {"id": "1", "text": "hello world", "category": "greeting", "length": 11},
        {
            "id": "2",
            "text": "goodbye cruel world",
            "category": "farewell",
            "length": 19,
        },
        {
            "id": "3",
            "text": "Python is awesome",
            "category": "programming",
            "length": 17,
        },
        {"id": "4", "text": "", "category": "empty", "length": 0},  # Edge case
    ]


@pytest.fixture
def create_jsonl_file(temp_workspace):
    """
    Factory fixture that creates JSONL files with given data.
    Returns a function that creates files.
    """

    def _create_file(filename: str, data: List[Dict[str, Any]]) -> Path:
        file_path = temp_workspace["data_dir"] / filename
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return file_path

    return _create_file


@pytest.fixture
def mock_data_loader():
    """Mock DataFileLoader for unit tests."""
    mock_loader = Mock()
    # Default behavior - can be overridden in tests
    mock_loader.load_input_data.return_value = {
        "1": {"text": "hello world", "category": "greeting"},
        "2": {"text": "goodbye world", "category": "farewell"},
    }
    return mock_loader

