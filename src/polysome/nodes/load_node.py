from typing import Optional, Dict, Any, List, Tuple, Set
from tqdm import tqdm
import logging
import json
from pathlib import Path
from polysome.utils.jsonl_writer import IncrementalJsonlWriter
from polysome.utils.data_loader import DataFileLoader
from polysome.nodes.node import (
    BaseNode,
    ValidationResult,
    node_step_error_handler,
)

logger = logging.getLogger(__name__)


class LoadNode(BaseNode):
    """
    A node that loads data from a supported file format (CSV, Excel, JSONL)
    using a primary key and writes it to a JSONL file.
    """

    def __init__(
        self,
        node_id: str,
        node_type: str,
        data_dir: Path,
        output_dir: Path,
        parent_wf_name: str,
        prompts_dir: Path,
        params: Dict[str, Any],
    ):
        super().__init__(
            node_id, node_type, parent_wf_name, data_dir, output_dir, prompts_dir, params
        )
        # Support both file-based (existing) and in-memory JSON (new GC) processing
        input_data_path_str = params.get("input_data_path")
        self.input_json_data = params.get("input_json_data", None)
        self.gc_mode = params.get("gc_mode", False)

        # Validation: Must have either input_data_path OR input_json_data
        if not input_data_path_str and self.input_json_data is None:
            raise ValueError("Either 'input_data_path' or 'input_json_data' must be specified in parameters.")
        
        if input_data_path_str and self.input_json_data is not None:
            raise ValueError("Cannot specify both 'input_data_path' and 'input_json_data'. Use one or the other.")

        self.primary_key = params.get("primary_key")
        if not self.primary_key:
            raise ValueError("'primary_key' must be specified in parameters.")

        self.data_attributes = params.get("data_attributes", None)
        
        # Resume functionality
        self.resume = params.get("resume", False)

        self.output_file_name = params.get(
            "output_file_name", f"{self.node_id}_output.jsonl"
        )
        self.output_data_path: Path = (
            self.output_dir / self.parent_wf_name / self.output_file_name
        )
        self.errors: List[Dict[str, Any]] = []
        self.status: str = "pending"
        self.output_info: Dict[str, Any] = {}
        self.data_loader: Optional[DataFileLoader] = None

        logger.info(
            f"Node '{self.node_id}' ({self.node_type}) initialized to load data and write to JSONL."
        )
        if self.input_json_data is not None:
            logger.info(f"  Mode: In-memory JSON processing (Grand Challenge mode: {self.gc_mode})")
            logger.info(f"  Input JSON data size: {len(str(self.input_json_data))} chars")
        else:
            logger.info(f"  Mode: File-based processing")
            logger.info(f"  Input data path: {self.input_data_path}")
        logger.info(f"  Primary Key: {self.primary_key}")
        logger.info(f"  Output JSONL path: {self.output_data_path}")

    def get_required_parameters(self) -> List[str]:
        """
        Return list of required parameter names for LoadNode.
        Note: Either input_data_path OR input_json_data is required, validated in constructor.
        """
        return ["primary_key"]

    def get_parameter_type_specs(self) -> Dict[str, type | Tuple[type, ...]]:
        """
        Return parameter type specifications for LoadNode.
        """
        return {
            "input_data_path": (str, Path),  # Accept both string and Path (optional)
            "input_json_data": dict,  # NEW: Accept dict for in-memory processing (optional)
            "gc_mode": bool,  # NEW: Grand Challenge mode flag (optional)
            "primary_key": str,
            "data_attributes": list,  # Should be a list if provided
            "output_file_name": str,
            "resume": bool,  # Inherited from BaseNode but specify type
        }

    def get_parameter_value_specs(self) -> Dict[str, Dict[str, Any]]:
        """
        Return parameter value specifications for LoadNode.
        """
        return {
            "primary_key": {
                "pattern": r"^[a-zA-Z_][a-zA-Z0-9_\-]*$",  # Valid identifier pattern (allows hyphens)
            },
            "output_file_name": {
                "pattern": r"^[\w\-. ]+\.jsonl$",  # Must end with .jsonl
            },
        }

    def _validate_custom_logic(self, result: ValidationResult) -> None:
        """
        Custom validation logic specific to LoadNode.
        """
        # Validate input source (file path OR in-memory JSON)
        if self.input_json_data is not None:
            # NEW: Validate in-memory JSON data
            self._validate_json_data(result)
        elif self.input_data_path:
            # EXISTING: Validate input file path exists and is accessible
            input_path = Path(self.input_data_path)

            if not input_path.exists():
                result.add_error(
                    "input_file_not_found",
                    f"Input file does not exist: {input_path}",
                    field="input_data_path",
                    value=str(input_path),
                )
            elif not input_path.is_file():
                result.add_error(
                    "input_not_file",
                    f"Input path is not a file: {input_path}",
                    field="input_data_path",
                    value=str(input_path),
                )
            else:
                # Check file format is supported
                supported_extensions = {".csv", ".xlsx", ".xls", ".jsonl", ".json"}
                if input_path.suffix.lower() not in supported_extensions:
                    result.add_warning(
                        "unsupported_file_format",
                        f"File format '{input_path.suffix}' may not be supported. "
                        f"Supported formats: {', '.join(supported_extensions)}",
                        field="input_data_path",
                        value=str(input_path),
                    )

                # Check file is not empty
                try:
                    if input_path.stat().st_size == 0:
                        result.add_warning(
                            "empty_input_file",
                            f"Input file appears to be empty: {input_path}",
                            field="input_data_path",
                            value=str(input_path),
                        )
                except OSError as e:
                    result.add_error(
                        "input_file_access_error",
                        f"Cannot access input file: {e}",
                        field="input_data_path",
                        value=str(input_path),
                    )

        # Validate that primary_key is not in conflict with reserved names
        reserved_names = {"_id", "_index", "_metadata"}
        if self.primary_key in reserved_names:
            result.add_warning(
                "reserved_primary_key",
                f"Primary key '{self.primary_key}' is a reserved name and may cause conflicts",
                field="primary_key",
                value=self.primary_key,
            )

    def _validate_json_data(self, result: ValidationResult) -> None:
        """
        NEW: Validate in-memory JSON data for Grand Challenge processing.
        """
        if not isinstance(self.input_json_data, dict):
            result.add_error(
                "invalid_json_data_type",
                f"input_json_data must be a dictionary, got {type(self.input_json_data)}",
                field="input_json_data",
                value=str(type(self.input_json_data)),
            )
            return

        # Check if primary key is present in the JSON data
        if self.primary_key not in self.input_json_data:
            result.add_warning(
                "missing_primary_key_in_json",
                f"Primary key '{self.primary_key}' not found in input JSON data. "
                "A default value will be used.",
                field="primary_key",
                value=self.primary_key,
            )

        # Validate JSON data is not empty
        if not self.input_json_data:
            result.add_warning(
                "empty_json_data",
                "Input JSON data is empty",
                field="input_json_data",
                value="{}",
            )

    def _validate_input_configuration(
        self, result: ValidationResult, input_data: Dict[str, Any] | None
    ) -> None:
        """
        Override input validation for LoadNode since it doesn't use dependencies.
        """
        # LoadNode is typically a source node and doesn't depend on other nodes
        if input_data:
            result.add_warning(
                "unexpected_dependencies",
                f"LoadNode typically doesn't have dependencies, but {len(input_data)} provided. "
                "These will be ignored.",
            )

        # LoadNode must have either input_data_path OR input_json_data configured
        if not self.input_data_path and self.input_json_data is None:
            result.add_error(
                "no_input_source",
                "LoadNode requires either input_data_path or input_json_data to be configured",
                field="input_data_path",
            )

    @node_step_error_handler(failure_status="failed_load_data")
    def _load_data(self) -> Dict[str, Dict[str, Any]]:
        """Loads data from either file or in-memory JSON based on configuration."""
        if not self.primary_key:
            raise ValueError(f"Node '{self.node_id}': primary_key cannot be empty.")

        if self.input_json_data is not None:
            # NEW: Process in-memory JSON for Grand Challenge
            return self._process_single_json_case(self.input_json_data)
        else:
            # EXISTING: Process file-based data (unchanged logic)
            if not self.input_data_path:
                raise ValueError(f"Node '{self.node_id}': input_data_path cannot be empty.")

            logger.info(
                f"Node '{self.node_id}': Attempting to load data from {self.input_data_path}..."
            )
            self.data_loader = DataFileLoader(
                input_data_path=self.input_data_path,
                primary_key=self.primary_key,
            )
            loaded_data = self.data_loader.load_input_data()
            logger.info(
                f"Node '{self.node_id}': Successfully loaded {len(loaded_data)} records."
            )
            return loaded_data

    def _process_single_json_case(self, json_data: dict) -> Dict[str, Dict[str, Any]]:
        """
        NEW: Process single JSON case for Grand Challenge mode.
        
        Args:
            json_data: Dictionary containing the case data
            
        Returns:
            Dictionary with primary_key as key and case data as value
        """
        # Extract the primary key value from the JSON data, or generate a default
        if self.primary_key in json_data:
            case_id = str(json_data[self.primary_key])
        else:
            # Generate a default case ID if primary key is not found
            case_id = "gc_case_001"
            logger.warning(
                f"Node '{self.node_id}': Primary key '{self.primary_key}' not found in JSON data. "
                f"Using default case ID: {case_id}"
            )

        logger.info(
            f"Node '{self.node_id}': Processing single JSON case with ID: {case_id}"
        )
        
        # Return in the same format as file-based loading
        return {case_id: json_data}

    @node_step_error_handler(failure_status="failed_write_jsonl")
    def _write_to_jsonl(self, data: Optional[Dict[str, Dict[str, Any]]]):
        """Writes the loaded data to a JSONL file."""
        if data is None:
            logger.warning(
                f"Node '{self.node_id}': No data provided (likely loading failed). Skipping JSONL writing."
            )
            return

        if self.primary_key is None:
            logger.error(
                f"Node '{self.node_id}': primary_key is None. Cannot write to JSONL."
            )
            raise ValueError(
                f"Node '{self.node_id}': primary_key became None unexpectedly."
            )

        self.output_data_path.parent.mkdir(parents=True, exist_ok=True)
        records_written = 0
        # Use append mode if resume is enabled, write mode otherwise
        write_mode = "a" if self.resume else "w"
        with IncrementalJsonlWriter(self.output_data_path, mode=write_mode) as writer:
            logger.info(
                f"Node '{self.node_id}': Writing {len(data)} items to {self.output_data_path}"
            )
            for key, row_data in tqdm(data.items(), desc=f"Writing {self.node_id}"):
                output_record = {self.primary_key: key}
                output_record.update(row_data)
                writer.write_row(output_record)
                records_written += 1

        logger.info(
            f"Node '{self.node_id}': Successfully wrote {records_written} records to {self.output_data_path}"
        )

    def _prepare_output_info(self, status: str, error_count: int) -> Dict[str, Any]:
        """Creates the standard output dictionary for the workflow."""
        return {
            "output_path": str(self.output_data_path),
            "output_attribute": None,
            "primary_key": self.primary_key,
            "status": status,
            "errors_count": error_count,
        }

    def _filter_data_attributes(self, loaded_data: Dict[str, Dict[str, Any]]) -> None:
        """Filters the loaded data based on specified data attributes."""
        if not self.data_attributes:
            return

        for key, row_data in loaded_data.items():
            # Keep only the specified attributes
            filtered_data = {
                attr: row_data[attr]
                for attr in self.data_attributes
                if attr in row_data
            }
            loaded_data[key] = filtered_data

    def _load_processed_ids(self) -> Set[str]:
        """Load already processed item IDs for resume functionality."""
        processed_ids = set()
        if not self.output_data_path.exists():
            return processed_ids

        logger.info(f"Node '{self.node_id}': Loading processed IDs for resume...")
        try:
            with open(self.output_data_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    # Progress logging for large files
                    if line_num % 5000 == 0:
                        logger.info(f"Node '{self.node_id}': Processing line {line_num} for resume...")
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict) and self.primary_key in data:
                            processed_ids.add(str(data[self.primary_key]))
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Node '{self.node_id}': Invalid JSON on line {line_num}, skipping"
                        )
        except Exception as e:
            logger.warning(f"Node '{self.node_id}': Error loading processed IDs: {e}")

        logger.info(
            f"Node '{self.node_id}': Found {len(processed_ids)} already processed items"
        )
        return processed_ids

    def run(self, input_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Executes the data loading and JSONL writing process."""
        logger.info(
            f"--- Starting run for Node '{self.node_id}' ({self.node_type}) ---"
        )
        self.errors = []
        self.status = "running"

        loaded_data = self._load_data()
        
        if not loaded_data:
            logger.warning(f"Node '{self.node_id}': No data loaded")
            self.status = "completed_no_new_items"
        else:
            total_items = len(loaded_data)
            logger.info(f"Node '{self.node_id}': Loaded {total_items} items")
            
            # Apply resume filtering if enabled
            logger.debug(f"Node '{self.node_id}': Checking resume flag: {self.resume}")
            if self.resume:
                logger.info(f"Node '{self.node_id}': Resume enabled, loading processed IDs...")
                processed_ids = self._load_processed_ids()
                logger.info(f"Node '{self.node_id}': Starting data filtering with {len(processed_ids)} processed IDs...")
                data_to_process = {
                    k: v for k, v in loaded_data.items() if str(k) not in processed_ids
                }
                logger.info(f"Node '{self.node_id}': Data filtering completed")
                skipped = total_items - len(data_to_process)
                if skipped > 0:
                    logger.info(
                        f"Node '{self.node_id}': Resume - skipping {skipped} already processed items"
                    )
                loaded_data = data_to_process
            else:
                logger.info(f"Node '{self.node_id}': Resume disabled, using all data")
            
            items_to_process = len(loaded_data)
            if items_to_process == 0:
                logger.info(f"Node '{self.node_id}': No items to process")
                self.status = "completed_no_new_items"
            else:
                logger.info(f"Node '{self.node_id}': {items_to_process} items selected for processing")

        if self.data_attributes and loaded_data:
            self._filter_data_attributes(loaded_data)

        if self.status == "running":
            if loaded_data:
                self._write_to_jsonl(loaded_data)
                # Determine final status based on whether writing caused errors (decorator handles errors list)
                if self.errors:
                    # Check if the status was already set to a failure state by decorators
                    if not self.status.startswith("failed_"):
                        self.status = "completed_with_errors"
                else:
                    # If no errors occurred during loading or writing
                    self.status = "completed_successfully"
            else:
                # No data to write, but that's okay for resume
                self.status = "completed_no_new_items"

        # If _load_data failed, self.status would already be set by the decorator
        elif not self.status.startswith("failed_") and self.status != "completed_no_new_items":
            logger.warning(
                f"Node '{self.node_id}': Skipping write to JSONL due to non-running status '{self.status}'."
            )

        error_count = len(self.errors)
        self.output_info = self._prepare_output_info(self.status, error_count)

        logger.info(
            f"Node '{self.node_id}' run finished with status: {self.status}, {error_count} errors."
        )
        logger.info(f"--- Finished run for Node '{self.node_id}' ---")
        return self.output_info
