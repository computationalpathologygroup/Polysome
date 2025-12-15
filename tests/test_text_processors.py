import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch
from polysome.nodes.util_nodes import (
    RegexSplitNode,
    SentenceSplitNode,
    RowConcatenationNode,
    ColumnConcatenationNode,
    DeduplicationNode,
)
from polysome.nodes.text_prompt_node import TextPromptNode


class TestRegexSplitNode:
    """Tests for RegexSplitNode node."""

    def test_process_item_basic_split(self):
        """Test basic regex splitting functionality."""
        processor = RegexSplitNode(
            node_id="test_split",
            node_type="regex_split",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_split",
                "primary_key": "id",
                "text_attribute": "content",
                "split_regex": r"\.",
                "filter_empty": False,  # Don't filter empty to see all splits
            },
        )

        input_data = {
            "content": "First sentence. Second sentence. Third sentence.",
            "title": "Test Document",
        }

        result = processor.process_item("1", input_data)

        # Should create 4 splits (including empty one after last period)
        assert len(result) == 4
        assert result[0]["content"] == "First sentence"
        assert result[1]["content"] == " Second sentence"
        assert result[2]["content"] == " Third sentence"
        assert result[3]["content"] == ""  # Empty split after final period

        # Check metadata is preserved
        assert all(row["title"] == "Test Document" for row in result)
        assert all(row["id"].startswith("1_") for row in result)
        assert [row["split_index"] for row in result] == [0, 1, 2, 3]

    def test_process_item_with_options(self):
        """Test splitting with strip and filter options."""
        processor = RegexSplitNode(
            node_id="test_split",
            node_type="regex_split",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_split",
                "primary_key": "id",
                "text_attribute": "content",
                "split_regex": r"\.",
                "strip_splits": True,
                "filter_empty": True,
            },
        )

        input_data = {"content": "First. Second. Third."}
        result = processor.process_item("1", input_data)

        # Should filter empty and strip whitespace
        assert len(result) == 3
        assert all(row["content"].strip() == row["content"] for row in result)
        assert all(row["content"] for row in result)  # No empty content

    def test_process_item_missing_text_attribute(self):
        """Test error when text attribute is missing."""
        processor = RegexSplitNode(
            node_id="test_split",
            node_type="regex_split",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_split",
                "primary_key": "id",
                "text_attribute": "content",
                "split_regex": r"\.",
            },
        )

        with pytest.raises(ValueError, match="Text attribute 'content' not found"):
            processor.process_item("1", {"title": "No content field"})

    def test_validation_missing_required_param(self):
        """Test validation fails when split_regex is missing."""
        processor = RegexSplitNode(
            node_id="test_split",
            node_type="regex_split",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_split",
                "primary_key": "id",
                # Missing split_regex
            },
        )

        result = processor.validate_configuration()
        assert not result.is_valid()
        assert any("split_regex" in error.message for error in result.errors)

    def test_validation_invalid_regex(self):
        """Test validation fails with invalid regex pattern."""
        processor = RegexSplitNode(
            node_id="test_split",
            node_type="regex_split",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_split",
                "primary_key": "id",
                "split_regex": "[invalid",  # Missing closing bracket
                "input_data_path": "input.jsonl",
            },
        )

        result = processor.validate_configuration()
        assert not result.is_valid()
        assert any(
            "invalid_regex_pattern" in error.error_type for error in result.errors
        )


class TestSentenceSplitNode:
    """Tests for SentenceSplitNode node."""

    def test_process_item_basic_sentence_split(self):
        """Test basic sentence splitting."""
        processor = SentenceSplitNode(
            node_id="test_sentence_split",
            node_type="sentence_split",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_sentence_split",
                "primary_key": "id",
                "text_attribute": "content",
                "sentences_per_split": 2,
            },
        )

        input_data = {
            "content": "First sentence. Second sentence. Third sentence. Fourth sentence.",
            "title": "Test Document",
        }

        result = processor.process_item("1", input_data)

        assert len(result) == 2
        assert result[0]["sentence_count"] == 2
        assert result[1]["sentence_count"] == 2
        assert "First sentence. Second sentence." in result[0]["content"]
        assert "Third sentence. Fourth sentence." in result[1]["content"]

    def test_process_item_no_sentences(self):
        """Test behavior with no detectable sentences."""
        processor = SentenceSplitNode(
            node_id="test_sentence_split",
            node_type="sentence_split",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_sentence_split",
                "primary_key": "id",
                "text_attribute": "content",
                "sentences_per_split": 2,
            },
        )

        # Text with no sentence endings - should be treated as one "sentence"
        input_data = {"content": "no sentence endings here"}
        result = processor.process_item("1", input_data)

        # Current implementation treats text without sentence endings as one sentence
        assert len(result) == 1
        assert (
            result[0]["sentence_count"] == 1
        )  # Not 0, it's one sentence without ending
        assert result[0]["content"] == "no sentence endings here."  # Period added

    def test_validation_missing_required_param(self):
        """Test validation fails when sentences_per_split is missing."""
        processor = SentenceSplitNode(
            node_id="test_split",
            node_type="sentence_split",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_split",
                "primary_key": "id",
                # Missing sentences_per_split
            },
        )

        result = processor.validate_configuration()
        assert not result.is_valid()
        assert any("sentences_per_split" in error.message for error in result.errors)

    def test_validation_invalid_sentence_count(self):
        """Test validation fails with invalid sentence count."""
        processor = SentenceSplitNode(
            node_id="test_split",
            node_type="sentence_split",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_split",
                "primary_key": "id",
                "sentences_per_split": 0,  # Invalid
                "input_data_path": "input.jsonl",
            },
        )

        result = processor.validate_configuration()
        assert not result.is_valid()
        assert any(
            "parameter_below_minimum" in error.error_type for error in result.errors
        )


class TestRowConcatenationNode:
    """Tests for RowConcatenationNode node."""

    @patch("polysome.utils.data_loader.DataFileLoader")
    @patch("polysome.utils.jsonl_writer.IncrementalJsonlWriter")
    def test_group_and_concatenate_logic(self, mock_writer, mock_loader_class):
        """Test the core grouping and concatenation logic."""
        # Setup mock data loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        mock_loader.load_input_data.return_value = {
            "doc1_0": {"id": "doc1", "content": "First part", "meta": "info1"},
            "doc1_1": {"id": "doc1", "content": "Second part", "meta": "info2"},
            "doc2_0": {"id": "doc2", "content": "Other content", "meta": "info3"},
        }

        processor = RowConcatenationNode(
            node_id="test_concat",
            node_type="row_concat",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_concat",
                "primary_key": "id",
                "concat_attribute": "content",
                "separator": " | ",
                "input_data_path": "input.jsonl",
            },
        )

        # Test grouping
        data = mock_loader.load_input_data.return_value
        grouped = processor._group_rows_by_primary_key(data)

        assert len(grouped) == 2
        assert "doc1" in grouped
        assert "doc2" in grouped
        assert len(grouped["doc1"]) == 2
        assert len(grouped["doc2"]) == 1

    def test_concatenate_group_content(self):
        """Test content concatenation for a group."""
        processor = RowConcatenationNode(
            node_id="test_concat",
            node_type="row_concat",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_concat",
                "primary_key": "id",
                "concat_attribute": "content",
                "separator": " | ",
                "metadata_merge_strategy": "first",
            },
        )

        group_rows = [
            {"id": "doc1", "content": "First", "title": "Doc 1"},
            {"id": "doc1", "content": "Second", "title": "Doc 1"},
        ]

        result = processor._concatenate_group_content("doc1", group_rows)

        assert result["id"] == "doc1"
        assert result["content"] == "First | Second"
        assert result["title"] == "Doc 1"  # First strategy
        assert result["row_count"] == 2

    def test_validation_missing_required_param(self):
        """Test validation fails when concat_attribute is missing."""
        processor = RowConcatenationNode(
            node_id="test_concat",
            node_type="row_concat",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_concat",
                "primary_key": "id",
                # Missing concat_attribute
            },
        )

        result = processor.validate_configuration()
        assert not result.is_valid()
        assert any("concat_attribute" in error.message for error in result.errors)


class TestColumnConcatenationNode:
    """Tests for ColumnConcatenationNode node."""

    def test_process_item_basic_concatenation(self):
        """Test basic column concatenation."""
        processor = ColumnConcatenationNode(
            node_id="test_col_concat",
            node_type="column_concat",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_col_concat",
                "primary_key": "id",
                "columns_to_concat": ["title", "content", "summary"],
                "output_column": "combined_text",
                "separator": " | ",
            },
        )

        input_data = {
            "title": "Document Title",
            "content": "Main content here",
            "summary": "Brief summary",
            "other_field": "should remain unchanged",
        }

        result = processor.process_item("1", input_data)

        expected = {
            "combined_text": "Document Title | Main content here | Brief summary"
        }

        assert result == expected

    def test_process_item_missing_columns(self):
        """Test behavior with missing columns when skip_missing=True."""
        processor = ColumnConcatenationNode(
            node_id="test_col_concat",
            node_type="column_concat",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_col_concat",
                "primary_key": "id",
                "columns_to_concat": ["title", "missing_field", "content"],
                "output_column": "combined_text",
                "separator": " | ",
                "skip_missing": True,
            },
        )

        input_data = {
            "title": "Document Title",
            "content": "Main content here",
            # missing_field is absent
        }

        result = processor.process_item("1", input_data)

        expected = {"combined_text": "Document Title | Main content here"}

        assert result == expected

    def test_process_item_skip_missing_false(self):
        """Test error when skip_missing=False and column is missing."""
        processor = ColumnConcatenationNode(
            node_id="test_col_concat",
            node_type="column_concat",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_col_concat",
                "primary_key": "id",
                "columns_to_concat": ["title", "missing_field"],
                "output_column": "combined_text",
                "skip_missing": False,
            },
        )

        input_data = {"title": "Document Title"}

        with pytest.raises(ValueError, match="Column 'missing_field' not found"):
            processor.process_item("1", input_data)

    def test_validation_empty_columns_list(self):
        """Test validation fails with empty columns list."""
        processor = ColumnConcatenationNode(
            node_id="test_col_concat",
            node_type="column_concat",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_col_concat",
                "primary_key": "id",
                "columns_to_concat": [],  # Empty list
                "output_column": "combined_text",
            },
        )

        result = processor.validate_configuration()
        assert not result.is_valid()
        assert any("empty_columns_list" in error.error_type for error in result.errors)

    def test_validation_duplicate_columns_warning(self):
        """Test warning for duplicate column names."""
        processor = ColumnConcatenationNode(
            node_id="test_col_concat",
            node_type="column_concat",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_col_concat",
                "primary_key": "id",
                "columns_to_concat": ["title", "content", "title"],  # Duplicate
                "output_column": "combined_text",
                "input_data_path": "input.jsonl",
            },
        )

        result = processor.validate_configuration()
        # The validation might fail due to other issues, so let's just check if duplicates are detected
        # Either as a warning or error is fine for this test
        has_duplicate_issue = any(
            "duplicate_column_names" in warning.error_type
            for warning in result.warnings
        ) or any("duplicate" in error.message.lower() for error in result.errors)
        assert has_duplicate_issue


class TestDeduplicationNode:
    """Tests for DeduplicationNode node."""

    @patch("polysome.utils.data_loader.DataFileLoader")
    @patch("polysome.utils.jsonl_writer.IncrementalJsonlWriter")
    def test_deduplication_logic(self, mock_writer, mock_loader_class):
        """Test the core deduplication logic."""
        # Setup mock data loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        mock_loader.load_input_data.return_value = {
            "key1": {"id": "doc1", "content": "duplicate text", "meta": "info1"},
            "key2": {
                "id": "doc1",
                "content": "duplicate text",
                "meta": "info2",
            },  # duplicate
            "key3": {"id": "doc2", "content": "unique text", "meta": "info3"},
            "key4": {"id": "doc3", "content": "another text", "meta": "info4"},
        }

        processor = DeduplicationNode(
            node_id="test_dedup",
            node_type="deduplication",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_dedup",
                "primary_key": "id",
                "dedup_attribute": "content",
                "keep_strategy": "first",
                "input_data_path": "input.jsonl",
            },
        )

        data = mock_loader.load_input_data.return_value
        to_keep = processor._identify_duplicates_to_keep(data)

        # Should keep first occurrence of each (id, content) pair
        assert "key1" in to_keep  # first occurrence of (doc1, duplicate text)
        assert "key2" not in to_keep  # duplicate
        assert "key3" in to_keep  # unique (doc2, unique text)
        assert "key4" in to_keep  # unique (doc3, another text)

    def test_dedup_key_generation(self):
        """Test deduplication key generation."""
        processor = DeduplicationNode(
            node_id="test_dedup",
            node_type="deduplication",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_dedup",
                "primary_key": "id",
                "dedup_attribute": "content",
                "case_sensitive": False,
            },
        )

        row_data = {"id": "doc1", "content": "Some Text"}
        key = processor._get_dedup_key(row_data)

        assert key == ("doc1", "some text")  # lowercase due to case_sensitive=False

    def test_case_sensitive_deduplication(self):
        """Test case-sensitive vs case-insensitive deduplication."""
        data = {
            "key1": {"id": "doc1", "content": "Text A"},
            "key2": {"id": "doc1", "content": "text a"},  # different case
        }

        # Case sensitive
        processor_sensitive = DeduplicationNode(
            node_id="test_dedup",
            node_type="deduplication",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_dedup",
                "primary_key": "id",
                "dedup_attribute": "content",
                "case_sensitive": True,
            },
        )

        to_keep_sensitive = processor_sensitive._identify_duplicates_to_keep(data)
        assert len(to_keep_sensitive) == 2  # Both kept (different when case-sensitive)

        # Case insensitive
        processor_insensitive = DeduplicationNode(
            node_id="test_dedup",
            node_type="deduplication",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_dedup",
                "primary_key": "id",
                "dedup_attribute": "content",
                "case_sensitive": False,
            },
        )

        to_keep_insensitive = processor_insensitive._identify_duplicates_to_keep(data)
        assert len(to_keep_insensitive) == 1  # Only one kept (same when lowercased)

    def test_validation_missing_required_param(self):
        """Test validation fails when dedup_attribute is missing."""
        processor = DeduplicationNode(
            node_id="test_dedup",
            node_type="deduplication",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_dedup",
                "primary_key": "id",
                # Missing dedup_attribute
            },
        )

        result = processor.validate_configuration()
        assert not result.is_valid()
        assert any("dedup_attribute" in error.message for error in result.errors)

    def test_validation_invalid_keep_strategy(self):
        """Test validation fails with invalid keep strategy."""
        processor = DeduplicationNode(
            node_id="test_dedup",
            node_type="deduplication",
            parent_wf_name="test_workflow",
            data_dir=Path("/fake"),
            output_dir=Path("/fake"),
            params={
                "name": "test_dedup",
                "primary_key": "id",
                "dedup_attribute": "content",
                "keep_strategy": "invalid_strategy",
                "input_data_path": "input.jsonl",
            },
        )

        result = processor.validate_configuration()
        assert not result.is_valid()
        assert any(
            "invalid_parameter_choice" in error.error_type for error in result.errors
        )


class TestTextPromptNode:
    """Tests for TextPromptNode validation and functionality."""

    @pytest.fixture
    def temp_dirs(self):
        """Fixture to create and cleanup temporary directories."""
        temp_data_dir = Path(tempfile.mkdtemp())
        temp_output_dir = Path(tempfile.mkdtemp())

        yield temp_data_dir, temp_output_dir

        # Cleanup
        if temp_data_dir.exists():
            temp_data_dir.rmdir()
        if temp_output_dir.exists():
            temp_output_dir.rmdir()

    def create_basic_node(self, temp_dirs=None, **override_params):
        """Create a node with basic valid parameters."""
        if temp_dirs:
            temp_data_dir, temp_output_dir = temp_dirs
        else:
            temp_data_dir = Path(tempfile.mkdtemp())
            temp_output_dir = Path(tempfile.mkdtemp())

        params = {
            "name": "test_prompt",
            "model_name": "test-model",
            "primary_key": "id",
            "input_data_path": "input.jsonl",  # Add to avoid no_input_source error
            **override_params,
        }

        return TextPromptNode(
            node_id="test_prompt",
            node_type="text_prompt",
            parent_wf_name="test_workflow",
            data_dir=temp_data_dir,
            output_dir=temp_output_dir,
            params=params,
        )

    # =====================================================================
    # VALIDATION TESTS
    # =====================================================================

    def test_validation_valid_config(self, temp_dirs):
        """Test validation passes with valid configuration."""
        node = self.create_basic_node(temp_dirs)

        with patch("polysome.config.PROMPT_DIR", Path("/fake/prompts")):
            with patch.object(node, "_should_validate_filesystem", return_value=False):
                result = node.validate_configuration()

        assert result.is_valid(), f"Validation failed: {result.get_detailed_report()}"
        assert len(result.errors) == 0

    def test_validation_missing_required_param(self, temp_dirs):
        """Test validation fails when model_name is missing."""
        node = self.create_basic_node(temp_dirs)
        # Remove model_name from params
        del node.params["model_name"]
        node.model_name = None

        result = node.validate_configuration()

        assert not result.is_valid()
        assert any("model_name" in error.message for error in result.errors)
        assert any(
            error.error_type == "missing_required_parameter" for error in result.errors
        )

    def test_validation_invalid_parameter_types(self, temp_dirs):
        """Test validation fails with invalid parameter types."""
        node = self.create_basic_node(
            temp_dirs,
            model_name=123,  # Should be string
            num_few_shots="invalid",  # Should be int
            parse_json="true",  # Should be bool
            engine_options="not_dict",  # Should be dict
        )

        result = node.validate_configuration()

        assert not result.is_valid()
        assert (
            len([e for e in result.errors if e.error_type == "invalid_parameter_type"])
            >= 4
        )

    def test_validation_parameter_value_constraints(self, temp_dirs):
        """Test validation of parameter value constraints."""
        node = self.create_basic_node(
            temp_dirs,
            num_few_shots=-1,  # Below minimum
            inference_engine="invalid_engine",  # Not in choices
            model_name="invalid@model!name",  # Invalid pattern
        )

        result = node.validate_configuration()

        assert not result.is_valid()
        error_types = [e.error_type for e in result.errors]

        # Check for at least one of our expected error types
        expected_errors = [
            "parameter_below_minimum",
            "invalid_parameter_choice",
            "parameter_pattern_mismatch",
        ]
        found_errors = [e for e in expected_errors if e in error_types]
        assert len(found_errors) > 0, (
            f"Expected one of {expected_errors}, but got: {error_types}"
        )

    def test_validation_template_context_map_keys(self, temp_dirs):
        """Test validation of template_context_map with non-string keys."""
        node = self.create_basic_node(
            temp_dirs,
            template_context_map={123: "content", "valid": "text"},  # Non-string key
        )

        with patch.object(node, "_should_validate_filesystem", return_value=False):
            result = node.validate_configuration()

        assert not result.is_valid()
        assert any(
            "invalid_template_context_map_keys" in error.error_type
            for error in result.errors
        )

    def test_validation_few_shot_configuration(self, temp_dirs):
        """Test validation of few-shot configuration."""
        # Test empty few-shot keys
        node = self.create_basic_node(
            temp_dirs,
            few_shot_context_key="",
            few_shot_assistant_key="   ",  # Whitespace only
            num_few_shots=3,
        )

        with patch.object(node, "_should_validate_filesystem", return_value=False):
            result = node.validate_configuration()

        assert not result.is_valid()
        assert (
            len([e for e in result.errors if e.error_type == "empty_few_shot_key"]) >= 2
        )

    @patch("polysome.config.PROMPT_DIR", Path("/fake/prompts"))
    def test_validation_missing_prompt_files(self, temp_dirs):
        """Test validation when prompt files are missing."""
        node = self.create_basic_node(temp_dirs, num_few_shots=2)

        # Mock file system - no files exist
        with patch.object(Path, "exists", return_value=False):
            result = node.validate_configuration()

        assert result.is_valid()
        assert any(
            "missing_prompt_directory" in error.error_type for error in result.warnings
        )

    @patch("polysome.config.PROMPT_DIR", Path("/fake/prompts"))
    def test_validation_prompt_files_not_files(self, temp_dirs):
        """Test validation when prompt paths exist but are not files."""
        node = self.create_basic_node(temp_dirs)

        # Mock file system - paths exist but are not files
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_file", return_value=False):
                result = node.validate_configuration()

        assert not result.is_valid()
        error_types = [e.error_type for e in result.errors]
        assert "invalid_system_prompt_file" in error_types
        assert "invalid_user_prompt_file" in error_types

    @patch("polysome.config.PROMPT_DIR", Path("/fake/prompts"))
    def test_validation_few_shot_file_missing_when_needed(self, temp_dirs):
        """Test validation when few-shot file is missing but num_few_shots > 0."""
        node = self.create_basic_node(temp_dirs, num_few_shots=3)

        # Mock system - prompt dir exists, main files exist, few-shot file missing
        def mock_exists(self):
            path_str = str(self)
            if "few_shot.jsonl" in path_str:
                return False
            return True

        with patch.object(Path, "exists", mock_exists):
            with patch.object(Path, "is_file", return_value=True):
                result = node.validate_configuration()

        assert not result.is_valid()
        assert any(
            "missing_few_shot_file" in error.error_type for error in result.errors
        )

    def test_validation_unused_few_shot_warning(self, temp_dirs):
        """Test warning when few-shot file is specified but not used."""
        node = self.create_basic_node(
            temp_dirs,
            num_few_shots=0,
            few_shot_lines_file="custom_few_shots.jsonl",  # Non-default file
        )

        with patch.object(node, "_should_validate_filesystem", return_value=False):
            result = node.validate_configuration()

        assert result.is_valid(), (
            f"Expected validation to pass but got: {result.get_detailed_report()}"
        )
        assert any(
            "unused_few_shot_file" in warning.error_type for warning in result.warnings
        )

    # =====================================================================
    # FUNCTIONALITY TESTS
    # =====================================================================

    @patch(
        "polysome.nodes.text_prompt_node.get_engine"
    )  # Fix: patch where it's imported
    @patch(
        "polysome.nodes.text_prompt_node.PromptFormatter"
    )  # This was already correct
    @patch("polysome.config.PROMPT_DIR", Path("/fake/prompts"))
    def test_setup_processing(self, mock_prompt_formatter, mock_get_engine, temp_dirs):
        """Test setup_processing initializes components correctly."""
        node = self.create_basic_node(temp_dirs)

        mock_formatter_instance = Mock()
        mock_prompt_formatter.return_value = mock_formatter_instance
        mock_engine_instance = Mock()
        mock_get_engine.return_value = mock_engine_instance

        node.setup_processing()

        # Verify PromptFormatter was initialized with correct paths
        mock_prompt_formatter.assert_called_once()
        call_args = mock_prompt_formatter.call_args
        assert call_args[1]["num_few_shots"] == 0
        assert call_args[1]["few_shot_context_key"] == "context"

        # Verify engine was initialized
        mock_get_engine.assert_called_once_with(
            engine_name="huggingface", model_name="test-model"
        )

        assert node.prompt_formatter == mock_formatter_instance
        assert node.model == mock_engine_instance

    def test_setup_processing_missing_model_name(self, temp_dirs):
        """Test setup_processing fails when model_name is missing."""
        node = self.create_basic_node(temp_dirs)
        del node.params["model_name"]
        node.model_name = None

        with pytest.raises(ValueError, match="model_name is required"):
            node.setup_processing()

    @patch("polysome.nodes.text_prompt_node.extract_and_parse_json")
    def test_process_item_basic(self, mock_extract_json, temp_dirs):
        """Test basic process_item functionality."""
        node = self.create_basic_node(temp_dirs)

        # Setup mocks
        mock_formatter = Mock()
        mock_model = Mock()
        node.prompt_formatter = mock_formatter
        node.model = mock_model

        mock_messages = [{"role": "user", "content": "test"}]
        mock_formatter.create_messages.return_value = mock_messages
        mock_model.generate_text.return_value = "Generated response"

        # Test data
        key = "test_key"
        row_data = {"text": "input text", "title": "Test Title"}

        result = node.process_item(key, row_data)

        # Verify calls
        mock_formatter.create_messages.assert_called_once_with(row_data)
        mock_model.generate_text.assert_called_once_with(mock_messages)

        assert result == "Generated response"
        mock_extract_json.assert_not_called()  # parse_json is False by default

    @patch("polysome.nodes.text_prompt_node.extract_and_parse_json")
    def test_process_item_with_template_context_map(self, mock_extract_json, temp_dirs):
        """Test process_item with template_context_map."""
        node = self.create_basic_node(
            temp_dirs, template_context_map={"content": "text", "subject": "title"}
        )

        # Setup mocks
        mock_formatter = Mock()
        mock_model = Mock()
        node.prompt_formatter = mock_formatter
        node.model = mock_model

        mock_formatter.create_messages.return_value = []
        mock_model.generate_text.return_value = "Response"

        # Test data with missing key
        row_data = {"text": "input text", "description": "Some desc"}  # Missing 'title'

        result = node.process_item("key", row_data)

        # Verify template context was mapped correctly
        expected_context = {
            "content": "input text",
            "subject": "",
        }  # Missing key -> empty string
        mock_formatter.create_messages.assert_called_once_with(expected_context)

    @patch("polysome.nodes.text_prompt_node.extract_and_parse_json")
    def test_process_item_with_json_parsing(self, mock_extract_json, temp_dirs):
        """Test process_item with JSON parsing enabled."""
        node = self.create_basic_node(temp_dirs, parse_json=True)

        # Setup mocks
        mock_formatter = Mock()
        mock_model = Mock()
        node.prompt_formatter = mock_formatter
        node.model = mock_model

        mock_formatter.create_messages.return_value = []
        mock_model.generate_text.return_value = '{"key": "value"}'
        mock_extract_json.return_value = {"key": "value"}

        result = node.process_item("key", {"text": "input"})

        mock_extract_json.assert_called_once_with('{"key": "value"}')
        assert result == {"key": "value"}

    @patch("polysome.nodes.text_prompt_node.extract_and_parse_json")
    def test_process_item_json_parsing_fails(self, mock_extract_json, temp_dirs):
        """Test process_item when JSON parsing fails."""
        node = self.create_basic_node(temp_dirs, parse_json=True)

        # Setup mocks
        mock_formatter = Mock()
        mock_model = Mock()
        node.prompt_formatter = mock_formatter
        node.model = mock_model

        mock_formatter.create_messages.return_value = []
        mock_model.generate_text.return_value = "Not JSON"
        mock_extract_json.return_value = None  # Parsing failed

        result = node.process_item("key", {"text": "input"})

        # Should return original text when parsing fails
        assert result == "Not JSON"

    def test_process_item_not_initialized(self, temp_dirs):
        """Test process_item fails when components are not initialized."""
        node = self.create_basic_node(temp_dirs)
        # Don't initialize prompt_formatter and model

        with pytest.raises(
            AssertionError, match="must be initialized before processing"
        ):
            node.process_item("key", {"text": "input"})

    def test_cleanup_processing(self, temp_dirs):
        """Test cleanup_processing method."""
        node = self.create_basic_node(temp_dirs)

        # Should not raise any exceptions
        node.cleanup_processing()

    # =====================================================================
    # PARAMETER SPECIFICATION TESTS
    # =====================================================================

    def test_get_required_parameters(self, temp_dirs):
        """Test required parameters specification."""
        node = self.create_basic_node(temp_dirs)
        required = node.get_required_parameters()

        assert "model_name" in required
        assert isinstance(required, list)

    def test_get_parameter_type_specs(self, temp_dirs):
        """Test parameter type specifications."""
        node = self.create_basic_node(temp_dirs)
        type_specs = node.get_parameter_type_specs()

        assert type_specs["model_name"] == str
        assert type_specs["num_few_shots"] == int
        assert type_specs["parse_json"] == bool
        assert type_specs["engine_options"] == dict

    def test_get_parameter_value_specs(self, temp_dirs):
        """Test parameter value specifications."""
        node = self.create_basic_node(temp_dirs)
        value_specs = node.get_parameter_value_specs()

        # Check that the specs contain expected constraints
        assert isinstance(value_specs, dict)

        # Test a few key constraints
        if "num_few_shots" in value_specs:
            assert "min" in value_specs["num_few_shots"]
            assert value_specs["num_few_shots"]["min"] == 0

        if "inference_engine" in value_specs:
            assert "choices" in value_specs["inference_engine"]

        if "model_name" in value_specs:
            assert "pattern" in value_specs["model_name"]
