from polysome.nodes.jsonl_processing_node import JSONLProcessingNode
from polysome.nodes.node import ValidationResult
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import re
from collections import defaultdict
import logging
from polysome.utils.jsonl_writer import IncrementalJsonlWriter

logger = logging.getLogger(__name__)


class RegexSplitNode(JSONLProcessingNode):
    """
    Node that splits text using regex patterns.

    Creates multiple output rows from a single input row based on regex splits.
    Each split becomes a new row with the same metadata but split content.
    """

    def __init__(
        self,
        node_id: str,
        node_type: str,
        parent_wf_name: str,
        data_dir: Path,
        output_dir: Path,
        params: Dict[str, Any],
    ):
        super().__init__(
            node_id, node_type, parent_wf_name, data_dir, output_dir, params
        )

        # Split configuration
        self.text_attribute = params.get("text_attribute", "text")
        self.split_regex = params.get("split_regex")

        # Processing options
        self.strip_splits = params.get("strip_splits", False)
        self.filter_empty = params.get("filter_empty", True)

    def get_required_parameters(self) -> List[str]:
        """Specify required parameters."""
        return ["split_regex"]

    def get_parameter_type_specs(self) -> Dict[str, type | Tuple[type, ...]]:
        """Specify parameter types."""
        return {
            "text_attribute": str,
            "split_regex": str,
            "strip_splits": bool,
            "filter_empty": bool,
        }

    def _validate_custom_logic(self, result: ValidationResult) -> None:
        """Custom validation for regex splitting."""
        # Validate regex pattern compilation (business logic)
        split_regex = self.params.get("split_regex")
        if (
            split_regex
        ):  # Only validate if present (required check handled by superclass)
            try:
                re.compile(split_regex)
            except re.error as e:
                result.add_error(
                    "invalid_regex_pattern",
                    f"Invalid regex pattern '{split_regex}': {e}",
                    field="split_regex",
                    value=split_regex,
                )

        # Warn about potentially problematic patterns
        if split_regex == "":
            result.add_warning(
                "empty_regex_pattern",
                "Empty regex pattern will not split text",
                field="split_regex",
            )

    def process_item(self, key: str, row_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split the text attribute using regex and create multiple rows.

        Returns a list of dictionaries, each representing a split row.
        """
        # Validate text attribute exists
        if self.text_attribute not in row_data:
            raise ValueError(
                f"Text attribute '{self.text_attribute}' not found in row data"
            )

        text_content = str(row_data[self.text_attribute])

        # Split using regex (compiled during validation if valid)
        assert self.split_regex, "split_regex must be defined"
        compiled_regex = re.compile(self.split_regex)
        splits = compiled_regex.split(text_content)

        # Apply processing options
        if self.strip_splits:
            splits = [s.strip() for s in splits]

        if self.filter_empty:
            splits = [s for s in splits if s]

        # Create output rows
        result_rows = []
        for i, split_content in enumerate(splits):
            # Create new row with split data
            new_row = row_data.copy()
            new_row[self.primary_key] = f"{key}_{i}"
            new_row[self.text_attribute] = split_content
            new_row["split_index"] = i

            result_rows.append(new_row)

        return result_rows


class SentenceSplitNode(JSONLProcessingNode):
    """
    Node that splits text into chunks containing a specified number of sentences.

    Creates multiple output rows from a single input row based on sentence grouping.
    Each chunk becomes a new row with the same metadata but chunked content.
    """

    def __init__(
        self,
        node_id: str,
        node_type: str,
        parent_wf_name: str,
        data_dir: Path,
        output_dir: Path,
        params: Dict[str, Any],
    ):
        super().__init__(
            node_id, node_type, parent_wf_name, data_dir, output_dir, params
        )

        # Split configuration
        self.text_attribute = params.get("text_attribute", "text")
        self.sentences_per_split = params.get("sentences_per_split", 1)
        self.sentence_endings = params.get("sentence_endings", r"[.!?]+")
        self.preserve_endings = params.get("preserve_endings", True)
        self.min_sentences_per_split = params.get("min_sentences_per_split", 1)

    def get_required_parameters(self) -> List[str]:
        """Specify required parameters."""
        return ["sentences_per_split"]

    def get_parameter_type_specs(self) -> Dict[str, type | Tuple[type, ...]]:
        """Specify parameter types."""
        return {
            "text_attribute": str,
            "sentences_per_split": int,
            "sentence_endings": str,
            "preserve_endings": bool,
            "min_sentences_per_split": int,
        }

    def get_parameter_value_specs(self) -> Dict[str, Dict[str, Any]]:
        """Specify parameter value constraints."""
        return {
            "sentences_per_split": {"min": 1},
            "min_sentences_per_split": {"min": 1},
        }

    def _validate_custom_logic(self, result: ValidationResult) -> None:
        """Custom validation for sentence splitting."""
        # Validate sentence endings regex
        if "sentence_endings" in self.params:
            try:
                re.compile(self.params["sentence_endings"])
            except re.error as e:
                result.add_error(
                    "invalid_sentence_endings_pattern",
                    f"Invalid sentence_endings pattern '{self.params['sentence_endings']}': {e}",
                    field="sentence_endings",
                    value=self.params["sentence_endings"],
                )

        # Validate logical constraints
        sentences_per = self.params.get("sentences_per_split", 1)
        min_sentences = self.params.get("min_sentences_per_split", 1)

        if min_sentences > sentences_per:
            result.add_warning(
                "min_sentences_exceeds_target",
                f"min_sentences_per_split ({min_sentences}) is greater than sentences_per_split ({sentences_per})",
                field="min_sentences_per_split",
            )

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences."""
        # Split on sentence endings

        sentence_pattern = re.compile(f"{self.sentence_endings}\\s*")
        parts = sentence_pattern.split(text.strip())

        # Filter out empty parts
        sentences = [s.strip() for s in parts if s.strip()]

        return sentences

    def _reconstruct_with_endings(
        self, sentences: List[str], original_text: str
    ) -> str:
        """Reconstruct sentence chunk with appropriate endings."""
        if not sentences:
            return ""

        if not self.preserve_endings:
            return " ".join(sentences)

        # Try to preserve original punctuation
        chunk_text = ". ".join(sentences)

        # Add final punctuation if missing
        if not chunk_text.endswith((".", "!", "?")):
            chunk_text += "."

        return chunk_text

    def process_item(self, key: str, row_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split the text attribute by sentences and create multiple rows.

        Returns a list of dictionaries, each representing a sentence chunk.
        """
        # Validate text attribute exists
        if self.text_attribute not in row_data:
            raise ValueError(
                f"Text attribute '{self.text_attribute}' not found in row data"
            )

        text_content = str(row_data[self.text_attribute])

        # Split into sentences
        sentences = self._split_into_sentences(text_content)

        if not sentences:
            # Return single row with original content if no sentences found
            new_row = row_data.copy()
            new_row[self.primary_key] = f"{key}_0"
            new_row["split_index"] = 0
            new_row["sentence_count"] = 0
            return [new_row]

        # Group sentences into chunks
        result_rows = []
        for i in range(0, len(sentences), self.sentences_per_split):
            chunk_sentences = sentences[i : i + self.sentences_per_split]

            # Skip if chunk is too small (unless it's the last chunk)
            if len(
                chunk_sentences
            ) < self.min_sentences_per_split and i + self.sentences_per_split < len(
                sentences
            ):
                continue

            # Reconstruct chunk text
            chunk_text = self._reconstruct_with_endings(chunk_sentences, text_content)

            # Create new row with chunk data
            new_row = row_data.copy()
            new_row[self.primary_key] = f"{key}_{len(result_rows)}"
            new_row[self.text_attribute] = chunk_text
            new_row["split_index"] = len(result_rows)
            new_row["sentence_count"] = len(chunk_sentences)
            new_row["sentence_start_index"] = i

            result_rows.append(new_row)

        return result_rows


class RowConcatenationNode(JSONLProcessingNode):
    """
    Node that concatenates rows with the same primary key.

    Groups all rows by primary key and concatenates specified attribute values,
    removing individual rows and creating single combined rows.
    """

    def __init__(
        self,
        node_id: str,
        node_type: str,
        parent_wf_name: str,
        data_dir: Path,
        output_dir: Path,
        params: Dict[str, Any],
    ):
        super().__init__(
            node_id, node_type, parent_wf_name, data_dir, output_dir, params
        )

        # Concatenation configuration
        self.concat_attribute = params.get("concat_attribute", "text")
        self.separator = params.get("separator", " ")
        self.sort_by_attribute = params.get("sort_by_attribute", None)
        self.metadata_merge_strategy = params.get("metadata_merge_strategy", "first")

    def get_required_parameters(self) -> List[str]:
        """Specify required parameters."""
        return ["concat_attribute"]

    def get_parameter_type_specs(self) -> Dict[str, type | Tuple[type, ...]]:
        """Specify parameter types."""
        return {
            "concat_attribute": str,
            "separator": str,
            "sort_by_attribute": (str, type(None)),
            "metadata_merge_strategy": str,
        }

    def get_parameter_value_specs(self) -> Dict[str, Dict[str, Any]]:
        """Specify parameter value constraints."""
        return {
            "metadata_merge_strategy": {"choices": ["first", "last", "most_common"]},
        }

    def _validate_custom_logic(self, result: ValidationResult) -> None:
        """Custom validation for row concatenation."""
        pass  # No additional validation needed

    def _group_rows_by_primary_key(
        self, data: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group rows by their primary key value."""
        groups = defaultdict(list)

        for _, row_data in data.items():
            primary_key_value = row_data.get(self.primary_key)
            if primary_key_value is not None:
                groups[str(primary_key_value)].append(row_data)

        return dict(groups)

    def _concatenate_group_content(
        self, group_key: str, group_rows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Concatenate content for a group of rows with the same primary key."""
        if not group_rows:
            return {}

        # Sort rows if specified
        if self.sort_by_attribute:
            try:
                # self.sort_by_attribute must be a string (and vaild attribute in rows), part of validation so casting is safe
                group_rows = sorted(
                    group_rows, key=lambda x: x.get(str(self.sort_by_attribute), 0)
                )
            except (TypeError, KeyError) as e:
                logger.warning(
                    f"Could not sort group {group_key} by {self.sort_by_attribute}: {e}"
                )

        # Extract content to concatenate
        content_parts = []
        for row in group_rows:
            content = row.get(self.concat_attribute, "")
            if content:  # Only add non-empty content
                content_parts.append(str(content))

        # Concatenate content
        concatenated_content = self.separator.join(content_parts)

        # Merge metadata based on strategy
        result_row = self._merge_metadata(group_rows)

        # Set the concatenated content and metadata
        result_row[self.primary_key] = group_key
        result_row[self.concat_attribute] = concatenated_content
        result_row["row_count"] = len(group_rows)

        return result_row

    def _merge_metadata(self, group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge metadata from multiple rows based on the configured strategy."""
        if not group_rows:
            return {}

        if self.metadata_merge_strategy == "first":
            return group_rows[0].copy()
        elif self.metadata_merge_strategy == "last":
            return group_rows[-1].copy()
        elif self.metadata_merge_strategy == "most_common":
            # For each field, find the most common value
            result = {}
            all_keys = set()
            for row in group_rows:
                all_keys.update(row.keys())

            for key in all_keys:
                if key == self.concat_attribute:
                    continue  # Skip the attribute we're concatenating

                values = [row.get(key) for row in group_rows if key in row]
                if values:
                    # Find most common value
                    value_counts = defaultdict(int)
                    for value in values:
                        value_counts[value] += 1
                    result[key] = max(value_counts.items(), key=lambda x: x[1])[0]

            return result

        return group_rows[0].copy()

    def run(self, input_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Override run method to handle row concatenation logic.
        This doesn't use the standard item-by-item processing.
        """
        logger.info(f"--- Starting RowConcatenationNode '{self.node_id}' ---")

        self.errors = []
        self.status = "running"

        try:
            # Resolve input and setup
            self._resolve_input(input_data)
            self._initialize_data_loader()
            self.setup_processing()

            if self.status != "running":
                return self._prepare_output_info(self.status, len(self.errors))

            # Load all data
            logger.info(
                f"Node '{self.node_id}': Loading data from {self.input_data_path}"
            )
            assert self.data_loader is not None, "Data loader must be initialized"
            all_data = self.data_loader.load_input_data()

            if not all_data:
                logger.warning(f"Node '{self.node_id}': No data to process")
                self.status = "completed_no_new_items"
                return self._prepare_output_info(self.status, len(self.errors))

            # Group rows by primary key
            grouped_data = self._group_rows_by_primary_key(all_data)
            logger.info(
                f"Node '{self.node_id}': Grouped {len(all_data)} rows into {len(grouped_data)} groups"
            )

            # Process each group and write results
            self.output_full_path.parent.mkdir(parents=True, exist_ok=True)

            with IncrementalJsonlWriter(self.output_full_path) as writer:
                for group_key, group_rows in grouped_data.items():
                    try:
                        concatenated_row = self._concatenate_group_content(
                            group_key, group_rows
                        )
                        if concatenated_row:
                            writer.write_row(concatenated_row)
                    except Exception as e:
                        error_entry = {
                            "key": group_key,
                            "error": str(e),
                            "type": type(e).__name__,
                        }
                        self.errors.append(error_entry)
                        logger.error(
                            f"Node '{self.node_id}': Error processing group {group_key}: {e}"
                        )

            # Set final status
            if self.errors:
                self.status = "completed_with_errors"
            else:
                self.status = "completed_successfully"

            logger.info(
                f"Node '{self.node_id}': Processed {len(grouped_data)} groups with {len(self.errors)} errors"
            )

        except Exception as e:
            logger.error(
                f"Node '{self.node_id}': Critical error during processing: {e}"
            )
            self.status = "failed_processing_execution"
            error_entry = {
                "key": "pipeline_error",
                "error": str(e),
                "type": type(e).__name__,
            }
            self.errors.append(error_entry)

        finally:
            try:
                self.cleanup_processing()
            except Exception as e:
                logger.warning(f"Node '{self.node_id}': Cleanup error (ignored): {e}")

        logger.info(f"--- Finished RowConcatenationNode '{self.node_id}' ---")
        return self._prepare_output_info(self.status, len(self.errors))

    def process_item(self, key: str, row_data: Dict[str, Any]) -> Any:
        """
        For compatibility, this method is not used in this node.
        """
        pass


class ColumnConcatenationNode(JSONLProcessingNode):
    """
    Node that concatenates multiple columns into a single new column.

    Takes specified column names and combines their text content with a separator.
    """

    def __init__(
        self,
        node_id: str,
        node_type: str,
        parent_wf_name: str,
        data_dir: Path,
        output_dir: Path,
        params: Dict[str, Any],
    ):
        super().__init__(
            node_id, node_type, parent_wf_name, data_dir, output_dir, params
        )

        # Concatenation configuration
        self.columns_to_concat = params.get("columns_to_concat", [])
        self.output_column = params.get("output_column", "concatenated_text")
        self.separator = params.get("separator", " ")
        self.skip_missing = params.get("skip_missing", True)
        self.skip_empty = params.get("skip_empty", True)

    def get_required_parameters(self) -> List[str]:
        """Specify required parameters."""
        return ["columns_to_concat"]

    def get_parameter_type_specs(self) -> Dict[str, type | Tuple[type, ...]]:
        """Specify parameter types."""
        return {
            "columns_to_concat": list,
            "output_column": str,
            "separator": str,
            "skip_missing": bool,
            "skip_empty": bool,
        }

    def _validate_custom_logic(self, result: ValidationResult) -> None:
        """Custom validation for column concatenation."""
        columns = self.params.get("columns_to_concat", [])

        # Validate columns list
        if not columns:
            result.add_error(
                "empty_columns_list",
                "columns_to_concat must contain at least one column name",
                field="columns_to_concat",
                value=columns,
            )

        # Check for duplicate column names
        if len(columns) != len(set(columns)):
            duplicates = [col for col in set(columns) if columns.count(col) > 1]
            result.add_warning(
                "duplicate_column_names",
                f"Duplicate column names found: {duplicates}",
                field="columns_to_concat",
            )

    def process_item(self, key: str, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Concatenate specified columns into a new column.

        Returns a dictionary with the new concatenated column.
        """
        content_parts = []

        for column_name in self.columns_to_concat:
            # Check if column exists
            if column_name not in row_data:
                if not self.skip_missing:
                    raise ValueError(f"Column '{column_name}' not found in row data")
                continue

            # Get column value
            column_value = row_data[column_name]

            # Convert to string and check if empty
            column_text = str(column_value) if column_value is not None else ""

            if self.skip_empty and not column_text.strip():
                continue

            content_parts.append(column_text)

        # Concatenate all parts
        concatenated_text = self.separator.join(content_parts)

        return {self.output_column: concatenated_text}


class DeduplicationNode(JSONLProcessingNode):
    """
    Node that removes duplicate rows based on primary key and a specified attribute.

    Identifies rows with the same primary key and deduplication attribute value,
    keeping only one instance based on the configured strategy.
    """

    def __init__(
        self,
        node_id: str,
        node_type: str,
        parent_wf_name: str,
        data_dir: Path,
        output_dir: Path,
        params: Dict[str, Any],
    ):
        super().__init__(
            node_id, node_type, parent_wf_name, data_dir, output_dir, params
        )

        # Deduplication configuration
        self.dedup_attribute = params.get("dedup_attribute", "text")
        self.keep_strategy = params.get("keep_strategy", "first")
        self.case_sensitive = params.get("case_sensitive", True)

    def get_required_parameters(self) -> List[str]:
        """Specify required parameters."""
        return ["dedup_attribute"]

    def get_parameter_type_specs(self) -> Dict[str, type | Tuple[type, ...]]:
        """Specify parameter types."""
        return {
            "dedup_attribute": str,
            "keep_strategy": str,
            "case_sensitive": bool,
        }

    def get_parameter_value_specs(self) -> Dict[str, Dict[str, Any]]:
        """Specify parameter value constraints."""
        return {
            "keep_strategy": {"choices": ["first", "last"]},
        }

    def _validate_custom_logic(self, result: ValidationResult) -> None:
        """Custom validation for deduplication."""
        # Validate dedup_attribute
        dedup_attr = self.params.get("dedup_attribute", "")

        # Check for potential conflicts with primary key
        if dedup_attr == self.primary_key:
            result.add_warning(
                "dedup_attribute_is_primary_key",
                f"dedup_attribute '{dedup_attr}' is the same as primary_key, this will deduplicate on primary key only",
                field="dedup_attribute",
            )

    def _get_dedup_key(self, row_data: Dict[str, Any]) -> Tuple[str, str]:
        """Generate a deduplication key from primary key and dedup attribute."""
        primary_key_value = str(row_data.get(self.primary_key, ""))
        dedup_value = str(row_data.get(self.dedup_attribute, ""))

        if not self.case_sensitive:
            dedup_value = dedup_value.lower()

        return (primary_key_value, dedup_value)

    def _identify_duplicates_to_keep(self, data: Dict[str, Any]) -> Set[str]:
        """Identify which row keys to keep based on deduplication strategy."""
        # Track occurrences of each (primary_key, dedup_attribute) combination
        dedup_groups = defaultdict(list)

        for row_key, row_data in data.items():
            dedup_key = self._get_dedup_key(row_data)
            dedup_groups[dedup_key].append(row_key)

        # Determine which rows to keep
        rows_to_keep = set()

        for dedup_key, row_keys in dedup_groups.items():
            if self.keep_strategy == "first":
                rows_to_keep.add(row_keys[0])
            elif self.keep_strategy == "last":
                rows_to_keep.add(row_keys[-1])

        return rows_to_keep

    def run(self, input_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Override run method to handle deduplication logic.
        This doesn't use the standard item-by-item processing.
        """
        logger.info(f"--- Starting DeduplicationNode '{self.node_id}' ---")

        self.errors = []
        self.status = "running"

        try:
            # Resolve input and setup
            self._resolve_input(input_data)
            self._initialize_data_loader()
            self.setup_processing()

            if self.status != "running":
                return self._prepare_output_info(self.status, len(self.errors))

            # Load all data
            logger.info(
                f"Node '{self.node_id}': Loading data from {self.input_data_path}"
            )
            assert self.data_loader is not None, "Data loader must be initialized"
            all_data = self.data_loader.load_input_data()

            if not all_data:
                logger.warning(f"Node '{self.node_id}': No data to process")
                self.status = "completed_no_new_items"
                return self._prepare_output_info(self.status, len(self.errors))

            original_count = len(all_data)

            # Identify rows to keep
            rows_to_keep = self._identify_duplicates_to_keep(all_data)
            duplicates_removed = original_count - len(rows_to_keep)

            logger.info(
                f"Node '{self.node_id}': Keeping {len(rows_to_keep)} out of {original_count} rows"
            )
            logger.info(
                f"Node '{self.node_id}': Removed {duplicates_removed} duplicate rows"
            )

            # Write deduplicated data
            self.output_full_path.parent.mkdir(parents=True, exist_ok=True)

            with IncrementalJsonlWriter(self.output_full_path) as writer:
                for row_key, row_data in all_data.items():
                    if row_key in rows_to_keep:
                        writer.write_row(row_data)

            # Set final status
            self.status = "completed_successfully"

            logger.info(f"Node '{self.node_id}': Deduplication completed successfully")

        except Exception as e:
            logger.error(
                f"Node '{self.node_id}': Critical error during processing: {e}"
            )
            self.status = "failed_processing_execution"
            error_entry = {
                "key": "pipeline_error",
                "error": str(e),
                "type": type(e).__name__,
            }
            self.errors.append(error_entry)

        finally:
            try:
                self.cleanup_processing()
            except Exception as e:
                logger.warning(f"Node '{self.node_id}': Cleanup error (ignored): {e}")

        logger.info(f"--- Finished DeduplicationNode '{self.node_id}' ---")
        return self._prepare_output_info(self.status, len(self.errors))

    def process_item(self, key: str, row_data: Dict[str, Any]) -> Any:
        """
        For compatibility, this method is not used in this node.
        """
        pass
