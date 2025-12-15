from polysome.nodes.jsonl_processing_node import JSONLProcessingNode
from polysome.nodes.node import ValidationResult, node_step_error_handler
from polysome.utils.data_loader import DataFileLoader
from polysome.utils.jsonl_writer import IncrementalJsonlWriter
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class CombineIntermediateOutputsNode(JSONLProcessingNode):
    """
    Node that combines outputs from multiple dependency nodes into a single JSONL file.
    
    Supports different join strategies (inner, left, outer) and configurable column mapping
    to handle conflicts between dependency outputs.
    """

    def __init__(
        self,
        node_id: str,
        node_type: str,
        parent_wf_name: str,
        data_dir: Path,
        output_dir: Path,
        prompts_dir: Path,
        params: Dict[str, Any],
    ):
        # Ensure 'name' parameter is set for base class compatibility
        if "name" not in params:
            params["name"] = f"combine_outputs_{node_id}"
        
        super().__init__(
            node_id, node_type, parent_wf_name, data_dir, output_dir, prompts_dir, params
        )

        # Combination configuration
        self.join_strategy = params.get("join_strategy", "inner")
        self.column_mapping = params.get("column_mapping", {})
        self.handle_conflicts = params.get("handle_conflicts", "error")
        self.retain_original_attributes = params.get("retain_original_attributes", False)
        
        # Will be populated during processing
        self.dependency_data_loaders: Dict[str, DataFileLoader] = {}
        self.dependency_output_info: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"CombineIntermediateOutputsNode '{self.node_id}' initialized")
        logger.info(f"  Join strategy: {self.join_strategy}")
        logger.info(f"  Column mapping: {self.column_mapping}")
        logger.info(f"  Conflict handling: {self.handle_conflicts}")
        logger.info(f"  Retain original attributes: {self.retain_original_attributes}")

    def get_required_parameters(self) -> List[str]:
        """Specify required parameters."""
        return []  # No strictly required parameters, but validation will check dependencies

    def get_parameter_type_specs(self) -> Dict[str, type | Tuple[type, ...]]:
        """Specify parameter types."""
        return {
            "join_strategy": str,
            "column_mapping": dict,
            "handle_conflicts": str,
            "retain_original_attributes": bool,
            "additional_output_formats": list,
            "output_format_options": dict,
        }

    def get_parameter_value_specs(self) -> Dict[str, Dict[str, Any]]:
        """Specify parameter value constraints."""
        return {
            "join_strategy": {"choices": ["inner", "left", "outer"]},
            "handle_conflicts": {"choices": ["error", "prefix_source", "suffix_source"]},
            "additional_output_formats": {
                "choices": ["excel", "json", "parquet"],  # Valid format options
            },
        }

    def _validate_custom_logic(self, result: ValidationResult) -> None:
        """Custom validation for combine outputs node."""
        # Validate join strategy
        join_strategy = self.params.get("join_strategy", "inner")
        if join_strategy not in ["inner", "left", "outer"]:
            result.add_error(
                "invalid_join_strategy",
                f"join_strategy must be one of ['inner', 'left', 'outer'], got '{join_strategy}'",
                field="join_strategy",
                value=join_strategy,
            )

        # Validate handle_conflicts setting
        handle_conflicts = self.params.get("handle_conflicts", "error")
        if handle_conflicts not in ["error", "prefix_source", "suffix_source"]:
            result.add_error(
                "invalid_handle_conflicts",
                f"handle_conflicts must be one of ['error', 'prefix_source', 'suffix_source'], got '{handle_conflicts}'",
                field="handle_conflicts",
                value=handle_conflicts,
            )

        # Validate column_mapping if provided
        column_mapping = self.params.get("column_mapping", {})
        if column_mapping:
            if not isinstance(column_mapping, dict):
                result.add_error(
                    "invalid_column_mapping_type",
                    f"column_mapping must be a dictionary, got {type(column_mapping).__name__}",
                    field="column_mapping",
                    value=str(column_mapping),
                )
            else:
                # Check for empty values in mapping
                for dep_id, column_name in column_mapping.items():
                    if not column_name or not isinstance(column_name, str):
                        result.add_error(
                            "invalid_column_mapping_value",
                            f"Column mapping for dependency '{dep_id}' must be a non-empty string, got '{column_name}'",
                            field="column_mapping",
                        )

                # Check for duplicate column names in mapping
                column_names = list(column_mapping.values())
                duplicate_columns = set([name for name in column_names if column_names.count(name) > 1])
                if duplicate_columns:
                    result.add_warning(
                        "duplicate_column_names_in_mapping",
                        f"Duplicate column names in mapping: {duplicate_columns}. This may cause conflicts.",
                        field="column_mapping",
                    )

        # Validate additional output formats
        additional_formats = self.params.get("additional_output_formats", [])
        if additional_formats:
            if not isinstance(additional_formats, list):
                result.add_error(
                    "invalid_additional_formats_type",
                    f"additional_output_formats must be a list, got {type(additional_formats).__name__}",
                    field="additional_output_formats",
                )
            else:
                valid_formats = ["excel", "json", "parquet"]
                for format_name in additional_formats:
                    if format_name not in valid_formats:
                        result.add_error(
                            "invalid_output_format",
                            f"Invalid output format '{format_name}'. Valid formats: {valid_formats}",
                            field="additional_output_formats",
                        )
                        
                # Check for duplicate formats
                if len(additional_formats) != len(set(additional_formats)):
                    duplicates = [fmt for fmt in set(additional_formats) if additional_formats.count(fmt) > 1]
                    result.add_warning(
                        "duplicate_output_formats",
                        f"Duplicate output formats: {duplicates}",
                        field="additional_output_formats",
                    )

        # Validate output format options
        format_options = self.params.get("output_format_options", {})
        if format_options and not isinstance(format_options, dict):
            result.add_error(
                "invalid_format_options_type",
                f"output_format_options must be a dictionary, got {type(format_options).__name__}",
                field="output_format_options",
            )

        # Validate retain_original_attributes
        retain_attrs = self.params.get("retain_original_attributes", False)
        if not isinstance(retain_attrs, bool):
            result.add_error(
                "invalid_retain_attributes_type",
                f"retain_original_attributes must be a boolean, got {type(retain_attrs).__name__}",
                field="retain_original_attributes",
            )

    def _validate_input_configuration(
        self, result: ValidationResult, input_data: Dict[str, Any] | None
    ) -> None:
        """Validate that we have multiple dependencies."""
        if not input_data:
            result.add_error(
                "no_dependencies",
                "CombineIntermediateOutputsNode requires at least 2 dependencies",
            )
            return

        if len(input_data) < 2:
            result.add_error(
                "insufficient_dependencies",
                f"CombineIntermediateOutputsNode requires at least 2 dependencies, got {len(input_data)}",
            )

        # Validate column mapping covers all dependencies if provided
        if self.column_mapping:
            missing_deps = set(input_data.keys()) - set(self.column_mapping.keys())
            if missing_deps:
                result.add_error(
                    "incomplete_column_mapping",
                    f"Column mapping missing for dependencies: {missing_deps}",
                    field="column_mapping",
                )

        # Validate dependency output structures
        for dep_id, dep_output in input_data.items():
            if not isinstance(dep_output, dict):
                result.add_error(
                    "invalid_dependency_output",
                    f"Dependency '{dep_id}' output is not a dictionary",
                )
                continue

            if "output_path" not in dep_output:
                result.add_error(
                    "missing_dependency_output_path",
                    f"Dependency '{dep_id}' missing 'output_path' in output",
                )

            if "primary_key" not in dep_output:
                result.add_error(
                    "missing_dependency_primary_key",
                    f"Dependency '{dep_id}' missing 'primary_key' in output",
                )

    def process_item(self, key: str, row_data: Dict[str, Any]) -> Any:
        """
        This method is not used in this node as we override the run method
        to handle the combination logic directly.
        """
        pass

    def run(self, input_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Main execution method for combining multiple dependency outputs.
        
        Overrides the standard JSONLProcessingNode run method to handle
        multiple input sources and combination logic.
        """
        logger.info(f"--- Starting CombineIntermediateOutputsNode '{self.node_id}' ---")

        self.errors = []
        self.status = "running"

        try:
            # Step 1: Resolve multiple inputs
            self._resolve_multiple_inputs(input_data)
            
            if self.status != "running":
                return self._prepare_output_info(self.status, len(self.errors))

            # Step 2: Load data from all dependencies
            dependency_data = self._load_dependency_data()
            
            if self.status != "running":
                return self._prepare_output_info(self.status, len(self.errors))

            # Step 3: Resolve column mapping and detect conflicts
            resolved_mapping = self._resolve_column_mapping()
            
            if self.status != "running":
                return self._prepare_output_info(self.status, len(self.errors))

            # Step 4: Perform join based on strategy
            combined_data = self._perform_join(dependency_data, resolved_mapping)
            
            if self.status != "running":
                return self._prepare_output_info(self.status, len(self.errors))

            # Step 5: Write combined output
            self._write_combined_output(combined_data)
            
            # Set final status
            if self.status == "running":
                if self.errors:
                    self.status = "completed_with_errors"
                else:
                    self.status = "completed_successfully"

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

        error_count = len(self.errors)
        logger.info(
            f"Node '{self.node_id}': Completed with {error_count} errors, status: {self.status}"
        )
        logger.info(f"--- Finished CombineIntermediateOutputsNode '{self.node_id}' ---")
        return self._prepare_output_info(self.status, error_count)

    @node_step_error_handler(failure_status="failed_resolve_multiple_inputs")
    def _resolve_multiple_inputs(self, input_data: Dict[str, Any] | None = None):
        """
        Resolve input data from multiple dependencies.
        
        Unlike the standard _resolve_input method, this handles multiple dependencies
        and stores their output information for later processing.
        """
        if not input_data:
            raise ValueError(f"Node '{self.node_id}': No input dependencies provided")

        if len(input_data) < 2:
            raise ValueError(
                f"Node '{self.node_id}': At least 2 dependencies required, got {len(input_data)}"
            )

        logger.info(
            f"Node '{self.node_id}': Resolving {len(input_data)} dependency inputs"
        )

        # Store dependency output info for each dependency
        for dep_id, dep_output in input_data.items():
            if not isinstance(dep_output, dict):
                raise ValueError(f"Dependency '{dep_id}' output is not a dictionary")

            if "output_path" not in dep_output:
                raise ValueError(f"Dependency '{dep_id}' missing 'output_path'")

            if "primary_key" not in dep_output:
                raise ValueError(f"Dependency '{dep_id}' missing 'primary_key'")

            self.dependency_output_info[dep_id] = dep_output.copy()
            
            logger.info(
                f"Node '{self.node_id}': Dependency '{dep_id}' -> {dep_output['output_path']}"
            )

        # Validate all dependencies use the same primary key
        primary_keys = {info["primary_key"] for info in self.dependency_output_info.values()}
        if len(primary_keys) > 1:
            raise ValueError(
                f"Node '{self.node_id}': All dependencies must use the same primary key. "
                f"Found: {primary_keys}"
            )

        # Set our primary key from the first dependency
        self.primary_key = next(iter(self.dependency_output_info.values()))["primary_key"]
        logger.info(f"Node '{self.node_id}': Using primary key: {self.primary_key}")

    @node_step_error_handler(failure_status="failed_load_dependency_data")
    def _load_dependency_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Load data from all dependency output files.
        
        Returns:
            Dict mapping dependency_id to loaded data dict
        """
        dependency_data = {}
        total_dependencies = len(self.dependency_output_info)
        
        logger.info(f"Node '{self.node_id}': Starting to load data from {total_dependencies} dependencies")
        
        for idx, (dep_id, dep_info) in enumerate(self.dependency_output_info.items(), 1):
            output_path = Path(dep_info["output_path"])
            primary_key = dep_info["primary_key"]
            
            logger.info(f"Node '{self.node_id}': [{idx}/{total_dependencies}] Loading data from '{dep_id}' at {output_path}")
            
            if not output_path.exists():
                raise FileNotFoundError(f"Dependency '{dep_id}' output file not found: {output_path}")

            # Create data loader for this dependency
            data_loader = DataFileLoader(
                input_data_path=output_path,
                primary_key=primary_key,
            )
            
            # Load the data
            loaded_data = data_loader.load_input_data()
            dependency_data[dep_id] = loaded_data
            
            logger.info(
                f"Node '{self.node_id}': [{idx}/{total_dependencies}] Loaded {len(loaded_data)} rows from dependency '{dep_id}'"
            )

        # Log overall statistics
        total_rows = sum(len(data) for data in dependency_data.values())
        logger.info(
            f"Node '{self.node_id}': Completed loading data: {total_rows} rows from {len(dependency_data)} dependencies"
        )
        
        return dependency_data

    @node_step_error_handler(failure_status="failed_load_original_input")
    def _load_original_input_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the original input data that was provided to all dependencies.
        
        This is used when retain_original_attributes=True to preserve all
        original data attributes in the combined output.
        
        Returns:
            Dict mapping primary key values to original input data
        """
        if not self.retain_original_attributes:
            return {}
            
        logger.info(f"Node '{self.node_id}': Loading original input data for attribute retention")
        
        # Find the original input source by looking for the dependency with input_data_path
        original_source_info = None
        for dep_id, dep_info in self.dependency_output_info.items():
            # Look for dependencies that have an input_data_path (indicating they load original data)
            if "input_data_path" in dep_info or dep_info.get("original_data_source"):
                original_source_info = dep_info
                break
                
        if not original_source_info:
            # If we can't find original source, use the first dependency as fallback
            logger.warning(f"Node '{self.node_id}': Could not identify original data source, using first dependency")
            dep_id, original_source_info = next(iter(self.dependency_output_info.items()))
            
        # Load the original data from the identified source
        original_data_path = Path(original_source_info["output_path"])
        primary_key = original_source_info["primary_key"]
        
        logger.info(f"Node '{self.node_id}': Loading original data from {original_data_path}")
        
        data_loader = DataFileLoader(
            input_data_path=original_data_path,
            primary_key=primary_key,
        )
        
        original_data = data_loader.load_input_data()
        logger.info(f"Node '{self.node_id}': Loaded {len(original_data)} original records")
        
        return original_data

    @node_step_error_handler(failure_status="failed_resolve_column_mapping")
    def _resolve_column_mapping(self) -> Dict[str, str]:
        """
        Resolve column mapping for each dependency.
        
        Returns:
            Dict mapping dependency_id to output column name
        """
        resolved_mapping = {}
        
        # If explicit mapping provided, use it
        if self.column_mapping:
            logger.info(f"Node '{self.node_id}': Using explicit column mapping")
            for dep_id in self.dependency_output_info.keys():
                if dep_id not in self.column_mapping:
                    raise ValueError(f"Column mapping missing for dependency '{dep_id}'")
                resolved_mapping[dep_id] = self.column_mapping[dep_id]
        else:
            # Use each dependency's output_attribute as column name
            logger.info(f"Node '{self.node_id}': Using automatic column mapping from output_attribute")
            for dep_id, dep_info in self.dependency_output_info.items():
                output_attribute = dep_info.get("output_attribute")
                if not output_attribute:
                    raise ValueError(
                        f"Dependency '{dep_id}' has no output_attribute and no explicit column mapping provided"
                    )
                resolved_mapping[dep_id] = output_attribute

        # Detect column name conflicts
        self._detect_column_conflicts(resolved_mapping)
        
        logger.info(f"Node '{self.node_id}': Resolved column mapping: {resolved_mapping}")
        return resolved_mapping

    def _detect_column_conflicts(self, resolved_mapping: Dict[str, str]) -> None:
        """
        Detect and handle column name conflicts based on handle_conflicts strategy.
        
        Args:
            resolved_mapping: Dict mapping dependency_id to column name (modified in place)
        """
        # Find duplicate column names
        column_counts = defaultdict(list)
        for dep_id, column_name in resolved_mapping.items():
            column_counts[column_name].append(dep_id)
        
        conflicts = {col: deps for col, deps in column_counts.items() if len(deps) > 1}
        
        if not conflicts:
            return  # No conflicts
        
        logger.warning(f"Node '{self.node_id}': Column name conflicts detected: {conflicts}")
        
        if self.handle_conflicts == "error":
            conflict_details = []
            for col, deps in conflicts.items():
                conflict_details.append(f"'{col}' used by {deps}")
            raise ValueError(
                f"Column name conflicts detected: {', '.join(conflict_details)}. "
                f"Use column_mapping or set handle_conflicts to 'prefix_source' or 'suffix_source'"
            )
        
        elif self.handle_conflicts == "prefix_source":
            for column_name, dep_ids in conflicts.items():
                for dep_id in dep_ids:
                    new_name = f"{dep_id}_{column_name}"
                    resolved_mapping[dep_id] = new_name
                    logger.info(f"Node '{self.node_id}': Renamed '{column_name}' -> '{new_name}' for dependency '{dep_id}'")
        
        elif self.handle_conflicts == "suffix_source":
            for column_name, dep_ids in conflicts.items():
                for dep_id in dep_ids:
                    new_name = f"{column_name}_{dep_id}"
                    resolved_mapping[dep_id] = new_name
                    logger.info(f"Node '{self.node_id}': Renamed '{column_name}' -> '{new_name}' for dependency '{dep_id}'")

    def _detect_additional_column_conflicts(
        self, 
        dependency_data: Dict[str, Dict[str, Any]], 
        resolved_mapping: Dict[str, str]
    ) -> None:
        """
        Detect conflicts between mapped columns and existing columns in the data.
        
        Args:
            dependency_data: Loaded data from all dependencies
            resolved_mapping: Resolved column mapping
        """
        # Get all column names from all dependencies
        all_existing_columns = set()
        for dep_id, data in dependency_data.items():
            if data:  # Check if data is not empty
                # Get column names from first row
                first_row = next(iter(data.values()))
                all_existing_columns.update(first_row.keys())
        
        # Check if any mapped column names conflict with existing columns
        mapped_columns = set(resolved_mapping.values())
        conflicts = mapped_columns.intersection(all_existing_columns)
        
        if conflicts:
            logger.warning(
                f"Node '{self.node_id}': Mapped column names conflict with existing columns: {conflicts}"
            )
            if self.handle_conflicts == "error":
                raise ValueError(
                    f"Mapped column names conflict with existing data columns: {conflicts}. "
                    f"Choose different column names or change handle_conflicts setting."
                )

    @node_step_error_handler(failure_status="failed_perform_join")
    def _perform_join(
        self, 
        dependency_data: Dict[str, Dict[str, Any]], 
        resolved_mapping: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform join operation based on configured strategy.
        
        Args:
            dependency_data: Loaded data from all dependencies
            resolved_mapping: Resolved column mapping
            
        Returns:
            Combined data ready for output
        """
        # Check for additional column conflicts between mapped names and existing data
        self._detect_additional_column_conflicts(dependency_data, resolved_mapping)
        
        if self.join_strategy == "inner":
            return self._perform_inner_join(dependency_data, resolved_mapping)
        elif self.join_strategy == "left":
            return self._perform_left_join(dependency_data, resolved_mapping)
        elif self.join_strategy == "outer":
            return self._perform_outer_join(dependency_data, resolved_mapping)
        else:
            raise ValueError(f"Unknown join strategy: {self.join_strategy}")

    def _perform_inner_join(
        self, 
        dependency_data: Dict[str, Dict[str, Any]], 
        resolved_mapping: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform inner join - only include rows where primary key exists in ALL dependencies.
        Logs detailed information about data loss.
        """
        logger.info(f"Node '{self.node_id}': Performing inner join")
        
        # Track original counts for logging
        original_counts = {dep_id: len(data) for dep_id, data in dependency_data.items()}
        
        # Find intersection of primary keys across all dependencies
        all_key_sets = [set(data.keys()) for data in dependency_data.values()]
        common_keys = set.intersection(*all_key_sets) if all_key_sets else set()
        
        # Log data loss for each dependency
        for dep_id, data in dependency_data.items():
            original_count = len(data)
            remaining_count = len(common_keys)
            lost_count = original_count - remaining_count
            
            if lost_count > 0:
                logger.warning(
                    f"Node '{self.node_id}': Inner join lost {lost_count} rows from dependency '{dep_id}' "
                    f"({remaining_count}/{original_count} rows retained)"
                )
            else:
                logger.info(
                    f"Node '{self.node_id}': Inner join retained all {remaining_count} rows from dependency '{dep_id}'"
                )
        
        # Overall statistics
        total_original = sum(original_counts.values())
        total_result_rows = len(common_keys)
        
        logger.info(
            f"Node '{self.node_id}': Inner join result: {total_result_rows} rows from {len(dependency_data)} dependencies. "
            f"Total input rows: {total_original}, final rows: {total_result_rows}"
        )
        
        if total_result_rows == 0:
            logger.warning(f"Node '{self.node_id}': Inner join resulted in 0 rows - no common primary keys found")
        
        return self._combine_data_for_keys(dependency_data, common_keys, resolved_mapping)

    def _combine_data_for_keys(
        self, 
        dependency_data: Dict[str, Dict[str, Any]], 
        keys_to_include: Set[str],
        resolved_mapping: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Combine data for specified keys using the resolved column mapping.
        
        Args:
            dependency_data: Loaded data from all dependencies
            keys_to_include: Set of primary key values to include
            resolved_mapping: Mapping from dependency_id to output column name
            
        Returns:
            Combined data with mapped column names
        """
        combined_data = {}
        total_keys = len(keys_to_include)
        
        logger.info(f"Node '{self.node_id}': Starting to combine data for {total_keys} keys")
        
        # Load original input data if attribute retention is enabled
        original_data = {}
        if self.retain_original_attributes:
            original_data = self._load_original_input_data()
            logger.info(f"Node '{self.node_id}': Loaded {len(original_data)} original records for attribute retention")
        
        processed_count = 0
        log_interval = max(1, total_keys // 10)  # Log every 10% or at least every row
        
        for key in keys_to_include:
            processed_count += 1
            
            # Log progress periodically
            if processed_count % log_interval == 0 or processed_count == total_keys:
                logger.info(f"Node '{self.node_id}': Combined {processed_count}/{total_keys} rows ({processed_count/total_keys*100:.1f}%)")
            
            # Start with original data if available, otherwise just primary key
            if self.retain_original_attributes and key in original_data:
                combined_row = original_data[key].copy()  # Copy all original attributes
                # Ensure primary key is correct (in case it was modified in processing)
                combined_row[self.primary_key] = key
            else:
                combined_row = {self.primary_key: key}
            
            # Add mapped outputs from each dependency
            for dep_id, data in dependency_data.items():
                output_column = resolved_mapping[dep_id]
                
                if key in data:
                    dep_row = data[key]
                    
                    # Get the dependency's output attribute value
                    dep_output_attribute = self.dependency_output_info[dep_id].get("output_attribute")
                    if dep_output_attribute and dep_output_attribute in dep_row:
                        # Handle potential column conflicts
                        if output_column in combined_row and self.handle_conflicts == "error":
                            raise ValueError(
                                f"Column conflict: '{output_column}' exists in both original data and dependency '{dep_id}'. "
                                f"Use column_mapping or set handle_conflicts to resolve."
                            )
                        elif output_column in combined_row:
                            # Apply conflict resolution strategy
                            if self.handle_conflicts == "prefix_source":
                                output_column = f"{dep_id}_{output_column}"
                            elif self.handle_conflicts == "suffix_source":
                                output_column = f"{output_column}_{dep_id}"
                                
                        combined_row[output_column] = dep_row[dep_output_attribute]
                    else:
                        # If no specific output attribute, include the whole row data
                        # (excluding primary key to avoid duplication)
                        for col, val in dep_row.items():
                            if col != self.primary_key:
                                prefixed_col = f"{dep_id}_{col}"
                                if prefixed_col in combined_row and self.handle_conflicts == "error":
                                    raise ValueError(
                                        f"Column conflict: '{prefixed_col}' already exists. "
                                        f"Use explicit column_mapping or change handle_conflicts setting."
                                    )
                                combined_row[prefixed_col] = val
                else:
                    # Key not found in this dependency - fill with null for left/outer joins
                    if output_column in combined_row and self.handle_conflicts == "error":
                        raise ValueError(
                            f"Column conflict: '{output_column}' exists in original data. "
                            f"Use explicit column_mapping to avoid conflicts."
                        )
                    elif output_column in combined_row:
                        # Apply conflict resolution for null values too
                        if self.handle_conflicts == "prefix_source":
                            output_column = f"{dep_id}_{output_column}"
                        elif self.handle_conflicts == "suffix_source":
                            output_column = f"{output_column}_{dep_id}"
                            
                    combined_row[output_column] = None
                
            combined_data[key] = combined_row
        
        logger.info(f"Node '{self.node_id}': Completed data combination for all {total_keys} keys")
        return combined_data

    def _perform_left_join(
        self, 
        dependency_data: Dict[str, Dict[str, Any]], 
        resolved_mapping: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform left join - include all rows from first dependency, fill missing with nulls.
        """
        logger.info(f"Node '{self.node_id}': Performing left join")
        
        # Use the first dependency as the "left" table
        dep_ids = list(dependency_data.keys())
        left_dep_id = dep_ids[0]
        left_data = dependency_data[left_dep_id]
        
        logger.info(f"Node '{self.node_id}': Using '{left_dep_id}' as left table ({len(left_data)} rows)")
        
        # Track statistics for logging
        left_keys = set(left_data.keys())
        null_counts = {dep_id: 0 for dep_id in dep_ids[1:]}  # Skip left dependency
        
        # For each other dependency, count how many keys are missing
        for dep_id in dep_ids[1:]:
            dep_keys = set(dependency_data[dep_id].keys())
            missing_keys = left_keys - dep_keys
            null_counts[dep_id] = len(missing_keys)
            
            if null_counts[dep_id] > 0:
                logger.info(
                    f"Node '{self.node_id}': Left join will fill {null_counts[dep_id]} null values "
                    f"for dependency '{dep_id}' ({len(dep_keys)}/{len(left_keys)} rows have data)"
                )
        
        total_nulls = sum(null_counts.values())
        logger.info(
            f"Node '{self.node_id}': Left join result: {len(left_keys)} rows, {total_nulls} null values filled"
        )
        
        return self._combine_data_for_keys(dependency_data, left_keys, resolved_mapping)

    def _perform_outer_join(
        self, 
        dependency_data: Dict[str, Dict[str, Any]], 
        resolved_mapping: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform outer join - include all rows from any dependency, fill missing with nulls.
        """
        logger.info(f"Node '{self.node_id}': Performing outer join")
        
        # Find union of all primary keys
        all_key_sets = [set(data.keys()) for data in dependency_data.values()]
        all_keys = set.union(*all_key_sets) if all_key_sets else set()
        
        # Track statistics for logging
        total_unique_keys = len(all_keys)
        null_counts = {}
        
        for dep_id, data in dependency_data.items():
            dep_keys = set(data.keys())
            missing_keys = all_keys - dep_keys
            null_counts[dep_id] = len(missing_keys)
            
            logger.info(
                f"Node '{self.node_id}': Dependency '{dep_id}' has data for {len(dep_keys)}/{total_unique_keys} rows "
                f"({null_counts[dep_id]} null values will be filled)"
            )
        
        total_nulls = sum(null_counts.values())
        total_data_points = total_unique_keys * len(dependency_data)
        
        logger.info(
            f"Node '{self.node_id}': Outer join result: {total_unique_keys} rows from {len(dependency_data)} dependencies. "
            f"Total data points: {total_data_points}, null values: {total_nulls}"
        )
        
        return self._combine_data_for_keys(dependency_data, all_keys, resolved_mapping)

    @node_step_error_handler(failure_status="failed_write_combined_output")
    def _write_combined_output(self, combined_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Write the combined data to the output JSONL file.
        
        Args:
            combined_data: Combined data ready for output
        """
        if not combined_data:
            logger.warning(f"Node '{self.node_id}': No combined data to write")
            return
        
        # Ensure output directory exists
        self.output_full_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Node '{self.node_id}': Writing {len(combined_data)} combined rows to {self.output_full_path}"
        )
        
        with IncrementalJsonlWriter(self.output_full_path) as writer:
            for key, row_data in combined_data.items():
                writer.write_row(row_data)
        
        logger.info(f"Node '{self.node_id}': Successfully wrote combined output")
        
        # Generate additional output formats if requested
        self._generate_additional_output_formats()

    def _prepare_output_info(self, status: str, error_count: int) -> Dict[str, Any]:
        """
        Prepare the output info dictionary for CombineNode.
        
        Overrides the parent method to include additional output files if generated.
        """
        base_output_info = {
            "output_path": str(self.output_full_path),
            "output_attribute": self.output_data_attribute,
            "primary_key": self.primary_key,
            "status": status,
            "errors_count": error_count,
        }
        
        # Use the BaseNode method to include additional output files
        return self._update_output_info_with_additional_formats(base_output_info)