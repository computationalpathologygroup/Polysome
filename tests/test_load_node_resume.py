import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from polysome.nodes.load_node import LoadNode
from polysome.utils.data_loader import DataFileLoader


class TestLoadNodeResume:
    """Test suite for LoadNode resume functionality."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        data_dir = temp_dir / "data"
        output_dir = temp_dir / "output"
        prompts_dir = temp_dir / "prompts"
        
        data_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)  
        prompts_dir.mkdir(parents=True)
        
        yield {
            "temp_dir": temp_dir,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "prompts_dir": prompts_dir
        }
        
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return {
            "item1": {"name": "Alice", "age": 30},
            "item2": {"name": "Bob", "age": 25},
            "item3": {"name": "Charlie", "age": 35},
            "item4": {"name": "Diana", "age": 28},
            "item5": {"name": "Eve", "age": 32}
        }

    def create_load_node(self, temp_dirs, resume=False):
        """Helper to create a LoadNode instance."""
        params = {
            "name": "test_load_node",
            "input_data_path": "input.json",  # Relative to data_dir
            "primary_key": "id",
            "resume": resume
        }
        
        return LoadNode(
            node_id="test_load_node",
            node_type="load_node",
            data_dir=temp_dirs["data_dir"],
            output_dir=temp_dirs["output_dir"],
            parent_wf_name="test_workflow",
            prompts_dir=temp_dirs["prompts_dir"],
            params=params
        )

    def write_existing_output(self, output_path, existing_data):
        """Helper to write existing output data."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item_id, data in existing_data.items():
                record = {"id": item_id}
                record.update(data)
                f.write(json.dumps(record) + "\n")

    @patch('polysome.nodes.load_node.DataFileLoader')
    def test_resume_no_existing_output(self, mock_loader_class, temp_dirs, sample_data):
        """Test resume functionality when no existing output file exists."""
        # Setup mock
        mock_loader = Mock()
        mock_loader.load_input_data.return_value = sample_data
        mock_loader_class.return_value = mock_loader

        # Create node with resume enabled
        node = self.create_load_node(temp_dirs, resume=True)
        
        # Run the node
        result = node.run()
        
        # Verify all items were processed (no existing file to resume from)
        assert result["status"] == "completed_successfully"
        
        # Check output file contains all items  
        output_path = temp_dirs["output_dir"] / "test_workflow" / "test_load_node_output.jsonl"
        assert output_path.exists()
        
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        assert len(lines) == 5  # All items should be written
        
        # Verify content
        written_ids = []
        for line in lines:
            data = json.loads(line.strip())
            written_ids.append(data["id"])
        
        assert set(written_ids) == set(sample_data.keys())

    @patch('polysome.nodes.load_node.DataFileLoader')
    def test_resume_with_partially_processed_data(self, mock_loader_class, temp_dirs, sample_data):
        """Test resume functionality with partially processed data."""
        # Setup mock
        mock_loader = Mock()
        mock_loader.load_input_data.return_value = sample_data
        mock_loader_class.return_value = mock_loader

        # Create existing output with some items already processed
        output_path = temp_dirs["output_dir"] / "test_workflow" / "test_load_node_output.jsonl"
        existing_data = {
            "item1": {"name": "Alice", "age": 30},
            "item3": {"name": "Charlie", "age": 35}
        }
        self.write_existing_output(output_path, existing_data)

        # Create node with resume enabled  
        node = self.create_load_node(temp_dirs, resume=True)
        
        # Run the node
        result = node.run()
        
        # Should complete successfully with only new items
        assert result["status"] == "completed_successfully"
        
        # Read final output
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Should have original 2 + new 3 = 5 total items
        assert len(lines) == 5
        
        # Verify all items are present (existing + new)
        all_ids = []
        for line in lines:
            data = json.loads(line.strip())
            all_ids.append(data["id"])
        
        assert set(all_ids) == set(sample_data.keys())

    @patch('polysome.nodes.load_node.DataFileLoader')  
    def test_resume_with_all_data_already_processed(self, mock_loader_class, temp_dirs, sample_data):
        """Test resume functionality when all data is already processed."""
        # Setup mock
        mock_loader = Mock()
        mock_loader.load_input_data.return_value = sample_data
        mock_loader_class.return_value = mock_loader

        # Create existing output with ALL items already processed
        output_path = temp_dirs["output_dir"] / "test_workflow" / "test_load_node_output.jsonl"
        self.write_existing_output(output_path, sample_data)

        # Create node with resume enabled
        node = self.create_load_node(temp_dirs, resume=True)
        
        # Run the node
        result = node.run()
        
        # Should complete with no new items
        assert result["status"] == "completed_no_new_items"
        
        # Read final output - should be unchanged
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Should still have exactly 5 items (no duplicates added)
        assert len(lines) == 5
        
        # Verify no duplicates were added
        all_ids = []
        for line in lines:
            data = json.loads(line.strip()) 
            all_ids.append(data["id"])
        
        # Should have exactly one of each ID
        assert len(all_ids) == len(set(all_ids))  # No duplicates
        assert set(all_ids) == set(sample_data.keys())

    @patch('polysome.nodes.load_node.DataFileLoader')
    def test_no_resume_overwrites_existing_file(self, mock_loader_class, temp_dirs, sample_data):
        """Test that without resume, existing output file is overwritten."""
        # Setup mock
        mock_loader = Mock()
        mock_loader.load_input_data.return_value = sample_data
        mock_loader_class.return_value = mock_loader

        # Create existing output file
        output_path = temp_dirs["output_dir"] / "test_workflow" / "test_load_node_output.jsonl"
        existing_data = {
            "old_item1": {"name": "OldAlice", "age": 99},
            "old_item2": {"name": "OldBob", "age": 99}
        }
        self.write_existing_output(output_path, existing_data)

        # Create node with resume DISABLED
        node = self.create_load_node(temp_dirs, resume=False)
        
        # Run the node
        result = node.run()
        
        assert result["status"] == "completed_successfully"
        
        # Read final output
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Should only have new data (old data overwritten)
        assert len(lines) == 5
        
        all_ids = []
        for line in lines:
            data = json.loads(line.strip())
            all_ids.append(data["id"])
        
        # Should only contain new sample data, not old data
        assert set(all_ids) == set(sample_data.keys())
        assert "old_item1" not in all_ids
        assert "old_item2" not in all_ids

    @patch('polysome.nodes.load_node.DataFileLoader')
    def test_resume_with_corrupted_output_file(self, mock_loader_class, temp_dirs, sample_data):
        """Test resume functionality handles corrupted output file gracefully."""
        # Setup mock
        mock_loader = Mock()
        mock_loader.load_input_data.return_value = sample_data
        mock_loader_class.return_value = mock_loader

        # Create corrupted output file (invalid JSON)
        output_path = temp_dirs["output_dir"] / "test_workflow" / "test_load_node_output.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write('{"id": "item1", "name": "Alice", "age": 30}\n')  # Valid
            f.write('invalid json line\n')  # Invalid
            f.write('{"id": "item2", "name": "Bob", "age": 25}\n')  # Valid

        # Create node with resume enabled
        node = self.create_load_node(temp_dirs, resume=True)
        
        # Run the node - should handle corruption gracefully
        result = node.run()
        
        assert result["status"] == "completed_successfully"
        
        # Should have processed items not in valid lines
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should contain all items (original valid ones + new ones)
        lines = content.strip().split('\n')
        valid_ids = set()
        
        for line in lines:
            try:
                data = json.loads(line)
                if "id" in data:
                    valid_ids.add(data["id"])
            except json.JSONDecodeError:
                pass  # Skip invalid lines
        
        # Should have all sample data IDs
        assert set(sample_data.keys()).issubset(valid_ids)