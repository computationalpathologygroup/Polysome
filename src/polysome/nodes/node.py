from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from functools import wraps
import logging
import functools
from pathlib import Path
import pandas as pd
import traceback
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error with context."""

    error_type: str
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    severity: str = "error"  # "error", "warning"


class ValidationResult:
    """Contains the results of validation checks."""

    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

    def add_error(
        self,
        error_type: str,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
    ):
        """Add a validation error."""
        self.errors.append(ValidationError(error_type, message, field, value, "error"))

    def add_warning(
        self,
        error_type: str,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
    ):
        """Add a validation warning."""
        self.warnings.append(
            ValidationError(error_type, message, field, value, "warning")
        )

    def is_valid(self) -> bool:
        """Check if validation passed (no errors, warnings are ok)."""
        return len(self.errors) == 0

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def get_summary(self) -> str:
        """Get a human-readable summary of validation results."""
        if self.is_valid() and not self.has_warnings():
            return "Validation passed"

        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")

        return f"Validation failed: {', '.join(parts)}"

    def get_detailed_report(self) -> str:
        """Get a detailed report of all validation issues."""
        lines = [self.get_summary()]

        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                field_str = f" (field: {error.field})" if error.field else ""
                lines.append(f"  - {error.error_type}: {error.message}{field_str}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                field_str = f" (field: {warning.field})" if warning.field else ""
                lines.append(f"  - {warning.error_type}: {warning.message}{field_str}")

        return "\n".join(lines)


class BaseNode:
    """Abstract base class for all workflow nodes."""

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
        self.node_id = node_id
        self.node_type = node_type
        self.parent_wf_name = parent_wf_name
        self.name = params["name"]
        self.primary_key = params.get("primary_key", None)
        self.errors = []
        self.status = "pending"  # Default status
        self.params = params

        # Input path and attribute are initially None; set during workflow run
        self.input_data_path: Optional[Path] = params.get("input_data_path")
        if self.input_data_path:
            self.input_data_path = Path(self.input_data_path)
            # merge base with input path
            self.input_data_path = data_dir / self.input_data_path
        self.data_attribute: Optional[str] = params.get("data_attribute")

        # Output path setup
        self.output_dir = output_dir
        self.output_data_path = output_dir / self.parent_wf_name
        self.output_data_file_name = f"{self.node_id}.jsonl"
        self.output_full_path = self.output_data_path / self.output_data_file_name

        # Additional output formats setup
        self.additional_output_formats = params.get("additional_output_formats", [])
        self.output_format_options = params.get("output_format_options", {})

        # Prompts directory setup
        self.prompts_dir = prompts_dir

        self.output_info: Dict[str, Any] = {}
        self.additional_output_files: Dict[str, Path] = {}
        logger.info(f"  Output will be saved to: {self.output_full_path}")

    @abstractmethod
    def run(self, input_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Executes the node's logic.

        Args:
            input_data: A dictionary where keys are dependency node IDs
                        and values are their output_info dictionaries.

        Returns:
            A dictionary containing information about the node's output
            (e.g., {'output_path': Path(...), 'output_attribute': '...', 'primary_key': '...'}).
            This structure will be passed to dependent nodes.
        """
        raise NotImplementedError("Subclasses must implement the run method.")

    def validate_configuration(
        self, input_data: Dict[str, Any] | None = None
    ) -> ValidationResult:
        """
        Comprehensive validation of node configuration.

        This method orchestrates all validation checks and returns a complete
        validation result. Override individual validation methods for custom logic.

        Args:
            input_data: Input data from dependencies (same as run() method)

        Returns:
            ValidationResult containing all validation errors and warnings
        """
        result = ValidationResult()

        try:
            # Core parameter validation
            self._validate_required_parameters(result)
            self._validate_parameter_types(result)
            self._validate_parameter_values(result)

            # Input/output validation
            self._validate_input_configuration(result, input_data)
            self._validate_output_configuration(result)

            # Custom validation (implemented by subclasses)
            self._validate_custom_logic(result)

            # File system validation (optional, might be expensive)
            if self._should_validate_filesystem():
                self._validate_filesystem_access(result)

        except Exception as e:
            result.add_error(
                "validation_framework_error",
                f"Unexpected error during validation: {e}",
            )
            logger.error(f"Node '{self.node_id}': Validation framework error: {e}")

        return result

    def _validate_required_parameters(self, result: ValidationResult) -> None:
        """Validate that all required parameters are present."""
        required_params = self.get_required_parameters()

        for param in required_params:
            if param not in self.params:
                result.add_error(
                    "missing_required_parameter",
                    f"Required parameter '{param}' is missing",
                    field=param,
                )
            elif self.params[param] is None:
                result.add_error(
                    "null_required_parameter",
                    f"Required parameter '{param}' is None",
                    field=param,
                    value=None,
                )

    def _validate_parameter_types(self, result: ValidationResult) -> None:
        """Validate that parameters have correct types."""
        type_specs = self.get_parameter_type_specs()
        for param, expected_type in type_specs.items():
            if param in self.params:
                value = self.params[param]
                if value is not None:
                    # Handle both single type and tuple of types
                    if isinstance(expected_type, tuple):
                        # Multiple types allowed
                        if not any(isinstance(value, t) for t in expected_type):
                            type_names = " or ".join(t.__name__ for t in expected_type)
                            result.add_error(
                                "invalid_parameter_type",
                                f"Parameter '{param}' must be of type {type_names}, got {type(value).__name__}",
                                field=param,
                                value=value,
                            )
                    else:
                        # Single type expected
                        if not isinstance(value, expected_type):
                            result.add_error(
                                "invalid_parameter_type",
                                f"Parameter '{param}' must be of type {expected_type.__name__}, got {type(value).__name__}",
                                field=param,
                                value=value,
                            )

    def _validate_parameter_values(self, result: ValidationResult) -> None:
        """Validate that parameter values are within acceptable ranges/choices."""
        value_specs = self.get_parameter_value_specs()

        for param, spec in value_specs.items():
            if param not in self.params:
                continue

            value = self.params[param]
            if value is None:
                continue

            # Check choices
            if "choices" in spec:
                if isinstance(value, list):
                    # For list parameters, check that each element is in choices
                    invalid_items = [
                        item for item in value if item not in spec["choices"]
                    ]
                    if invalid_items:
                        result.add_error(
                            "invalid_parameter_choice",
                            f"Parameter '{param}' contains invalid choices {invalid_items}. Valid choices: {spec['choices']}",
                            field=param,
                            value=value,
                        )
                else:
                    # For non-list parameters, check the value directly
                    if value not in spec["choices"]:
                        result.add_error(
                            "invalid_parameter_choice",
                            f"Parameter '{param}' must be one of {spec['choices']}, got {value}",
                            field=param,
                            value=value,
                        )

            # Check range
            if "min" in spec and value < spec["min"]:
                result.add_error(
                    "parameter_below_minimum",
                    f"Parameter '{param}' must be >= {spec['min']}, got {value}",
                    field=param,
                    value=value,
                )

            if "max" in spec and value > spec["max"]:
                result.add_error(
                    "parameter_above_maximum",
                    f"Parameter '{param}' must be <= {spec['max']}, got {value}",
                    field=param,
                    value=value,
                )

            # Check regex pattern
            if "pattern" in spec and isinstance(value, str):
                import re

                if not re.match(spec["pattern"], value):
                    result.add_error(
                        "parameter_pattern_mismatch",
                        f"Parameter '{param}' does not match required pattern {spec['pattern']}",
                        field=param,
                        value=value,
                    )

    def _validate_input_configuration(
        self, result: ValidationResult, input_data: Dict[str, Any] | None
    ) -> None:
        """Validate input data configuration."""
        try:
            # Try to resolve input (similar to _resolve_input but for validation)
            if input_data:
                if len(input_data) > 1:
                    result.add_warning(
                        "multiple_dependencies",
                        f"Node has {len(input_data)} dependencies, will use first one",
                    )

                dep_id, dep_output = next(iter(input_data.items()))

                # Check dependency output structure
                if not isinstance(dep_output, dict):
                    result.add_error(
                        "invalid_dependency_output",
                        f"Dependency '{dep_id}' output is not a dictionary",
                    )
                    return

                if "output_path" not in dep_output:
                    result.add_error(
                        "missing_dependency_output_path",
                        f"Dependency '{dep_id}' missing 'output_path' in output",
                    )

                # Validate inherited values
                resolved_path = dep_output.get("output_path")
                if resolved_path and Path(resolved_path).suffix not in [
                    ".jsonl",
                    ".json",
                ]:
                    result.add_warning(
                        "unexpected_input_format",
                        f"Input file format may not be supported: {Path(resolved_path).suffix}",
                    )

            elif not self.input_data_path:
                result.add_error(
                    "no_input_source",
                    "No input_data_path configured and no dependencies provided",
                )

            # Validate primary key will be available
            if not self.primary_key and not (
                input_data
                and any(dep.get("primary_key") for dep in input_data.values())
            ):
                result.add_error(
                    "no_primary_key",
                    "No primary_key configured and none available from dependencies",
                )

        except Exception as e:
            result.add_error(
                "input_validation_error",
                f"Error validating input configuration: {e}",
            )

    def _validate_output_configuration(self, result: ValidationResult) -> None:
        """Validate output configuration."""
        # Check output directory permissions (if parent exists)
        if self.output_dir.exists() and not self.output_dir.is_dir():
            result.add_error(
                "invalid_output_directory",
                f"Output path exists but is not a directory: {self.output_dir}",
            )

        # Check if we can create the output directory
        try:
            self.output_full_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            result.add_error(
                "output_permission_denied",
                f"Cannot create output directory: {self.output_full_path.parent}",
            )
        except Exception as e:
            result.add_error(
                "output_directory_error",
                f"Error with output directory: {e}",
            )

    def _validate_filesystem_access(self, result: ValidationResult) -> None:
        """Validate filesystem access (expensive checks)."""
        # Check input file accessibility if specified
        if self.input_data_path and self.input_data_path.exists():
            if not self.input_data_path.is_file():
                result.add_error(
                    "input_not_file",
                    f"Input path is not a file: {self.input_data_path}",
                )
            elif not self.input_data_path.stat().st_size > 0:
                result.add_warning(
                    "empty_input_file",
                    f"Input file appears to be empty: {self.input_data_path}",
                )

    def _validate_custom_logic(self, result: ValidationResult) -> None:
        """
        Custom validation logic for subclasses.

        Override this method to add node-specific validation.
        The base implementation does nothing.
        """
        pass

    def _should_validate_filesystem(self) -> bool:
        """
        Whether to perform expensive filesystem validation.

        Override to control when filesystem checks are performed.
        """
        return True

    # =====================================================================
    # VALIDATION SPECIFICATION METHODS (Override in subclasses)
    # =====================================================================

    def get_required_parameters(self) -> List[str]:
        """
        Return list of required parameter names.

        Override in subclasses to specify required parameters.
        """
        return []

    def get_parameter_type_specs(self) -> Dict[str, type | Tuple[type, ...]]:
        """
        Return parameter type specifications.
        Override in subclasses to specify parameter types.
        Returns:
            Dict mapping parameter names to expected types or lists of types
        """
        return {}

    def get_parameter_value_specs(self) -> Dict[str, Dict[str, Any]]:
        """
        Return parameter value specifications.

        Override in subclasses to specify parameter value constraints.

        Returns:
            Dict mapping parameter names to constraint specifications:
            {
                'param_name': {
                    'choices': [list of valid choices],
                    'min': minimum_value,
                    'max': maximum_value,
                    'pattern': regex_pattern_string
                }
            }
        """
        return {}

    def _generate_additional_output_formats(self) -> None:
        """
        Generate additional output formats from the primary JSONL file.

        This method is called after the primary JSONL file has been written
        to create additional formats like Excel, JSON array, and Parquet.
        """
        if not self.additional_output_formats:
            return  # No additional formats requested

        logger.info(
            f"Node '{self.node_id}': Generating {len(self.additional_output_formats)} additional output formats"
        )

        # Import here to avoid circular import and optional dependency issues
        try:
            from polysome.utils.output_formatter import OutputFormatter
        except ImportError as e:
            logger.error(f"Node '{self.node_id}': Cannot import OutputFormatter: {e}")
            return

        # Use the base name without extension for additional formats
        output_base_path = self.output_full_path.with_suffix("")

        try:
            generated_files = OutputFormatter.convert_jsonl_to_formats(
                jsonl_path=self.output_full_path,
                output_base_path=output_base_path,
                formats=self.additional_output_formats,
                format_options=self.output_format_options,
            )

            # Store the generated file paths
            self.additional_output_files.update(generated_files)

            logger.info(
                f"Node '{self.node_id}': Successfully generated additional formats: {list(generated_files.keys())}"
            )

        except Exception as e:
            logger.error(
                f"Node '{self.node_id}': Failed to generate additional output formats: {e}"
            )
            # Don't raise exception - additional formats are optional

    def _update_output_info_with_additional_formats(
        self, base_output_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update the output info dictionary to include additional output file paths.

        Args:
            base_output_info: Base output info dictionary

        Returns:
            Updated output info with additional formats included
        """
        updated_info = base_output_info.copy()

        if self.additional_output_files:
            # Add additional format file paths
            updated_info["additional_output_files"] = {
                format_name: str(file_path)
                for format_name, file_path in self.additional_output_files.items()
            }
            updated_info["additional_formats"] = list(
                self.additional_output_files.keys()
            )

        return updated_info

    def cleanup_processing(self) -> None:
        """
        Hook for cleanup after processing.

        This is a default implementation that does nothing. Subclasses can override
        this method to perform cleanup tasks like releasing resources, closing
        connections, unloading models, etc.

        This method is called at the end of processing regardless of success/failure.
        """
        pass


def processing_exception_handler(error_list_attr: str, key_arg_index: int):
    """
    Decorator for methods processing individual items within a loop.

    Catches exceptions during the execution of the wrapped function (presumably
    processing a single item), logs the error, appends error details (including
    the item's key) to a specified list attribute on the instance, and prevents
    the exception from propagating, allowing the calling loop to continue.

    Args:
        error_list_attr (str): The name of the list attribute on the instance
                                (self) where error dictionaries should be appended.
                                Example: "errors".
        key_arg_index (int): The positional argument index in the wrapped function's
                                signature that holds the unique identifier (key) for
                                the item being processed. Remember that 'self' is at index 0.
                                So, for a method `def process(self, key, value)`,
                                the key is at index 1.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            item_key = "<key_not_retrieved>"
            node_id_str = getattr(
                self, "node_id", "UNKNOWN_NODE"
            )  # Get node_id if available

            try:
                # --- Attempt to retrieve the item key for error reporting ---
                # Adjust index: 'self' is arg 0, so actual args start at 1 for the wrapped method
                actual_arg_position = key_arg_index - 1
                if actual_arg_position >= 0 and actual_arg_position < len(args):
                    item_key = args[actual_arg_position]
                else:
                    # Log a warning if the expected key argument wasn't found positionally
                    logger.warning(
                        f"Node '{node_id_str}': Decorator '{processing_exception_handler.__name__}' "
                        f"could not retrieve key at index {key_arg_index} from args for function '{func.__name__}'. "
                        f"Args provided: {args}. Using '{item_key}'."
                    )
            except Exception as key_retrieval_error:
                # This shouldn't typically happen, but catch errors during key retrieval itself
                logger.error(
                    f"Node '{node_id_str}': Error retrieving key in decorator for '{func.__name__}': {key_retrieval_error}"
                )

            try:
                # --- Execute the original item processing function ---
                result = func(self, *args, **kwargs)
                return result  # Return the result if successful

            except Exception as e:
                # --- Handle exceptions occurring within the wrapped function ---
                try:
                    # Retrieve the designated error list from the instance
                    error_list = getattr(self, error_list_attr)

                    # Verify it's actually a list
                    if not isinstance(error_list, list):
                        logger.error(
                            f"Node '{node_id_str}': Decorator configuration error for '{func.__name__}'. "
                            f"Attribute '{error_list_attr}' is not a list (found type {type(error_list)}). "
                            f"Cannot record error for key '{item_key}'."
                        )
                        # Cannot proceed with error recording, but still suppress the original exception 'e'
                        return None  # Indicate failure without stopping the loop

                    # Prepare the error details dictionary
                    error_details = {
                        "key": item_key,
                        "error": str(e),
                        "type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                    }

                    # Append the error details to the list
                    error_list.append(error_details)

                    # Log the specific item failure clearly
                    logger.error(
                        f"Node '{node_id_str}': Failed processing item with key '{item_key}'. Error: {e}",
                        exc_info=False,  # Set to True if you want the stack trace in the main log file too
                    )

                except AttributeError:
                    # Handle case where the instance doesn't have the specified error list attribute
                    logger.error(
                        f"Node '{node_id_str}': Decorator configuration error for '{func.__name__}'. "
                        f"Instance does not have an attribute named '{error_list_attr}'. "
                        f"Cannot record error for key '{item_key}'."
                    )
                except Exception as handler_error:
                    # Catch unexpected errors within the exception handler itself
                    logger.critical(
                        f"Node '{node_id_str}': CRITICAL error within exception handler decorator "
                        f"while handling error for key '{item_key}'. Handler error: {handler_error}",
                        exc_info=True,
                    )

                return None

        return wrapper

    return decorator


def node_step_error_handler(failure_status: str, log_level: int = logging.CRITICAL):
    """
    Decorator to handle exceptions during node setup or execution steps.

    Args:
        failure_status (str): The status string to set on the node if an exception occurs.
        log_level (int): The logging level for the caught exception.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self: "BaseNode", *args, **kwargs):
            try:
                # Execute the original method
                result = func(self, *args, **kwargs)
                # Return original result (e.g., data_to_process, None)
                return result
            except (
                ValueError,
                RuntimeError,
                FileNotFoundError,
                IOError,
                pd.errors.ParserError,
                KeyError,
                Exception,
            ) as e:
                logger.log(
                    log_level,
                    f"FAILURE in Node '{self.node_id}' during '{func.__name__}': {e}",
                    exc_info=True,
                )
                # Add error only if it's a new one for this run
                # (Prevents duplicate errors if caught at multiple levels, though we aim to catch once)
                error_entry = {
                    "key": f"Step-{func.__name__}",
                    "error": str(e),
                    "type": type(e).__name__,
                }
                if not hasattr(self, "errors"):
                    self.errors = []
                if error_entry not in self.errors:  # Basic check for duplicates
                    self.errors.append(error_entry)

                # Set the node's status attribute directly
                self.status = failure_status
                # Indicate failure - the caller (run method) will check self.status
                # For methods returning data, returning None or an empty structure might be appropriate
                # Or simply rely on the status check. Let's rely on status for now.
                return None  # Or False, or an empty version of expected output if needed by caller logic

        return wrapper

    return decorator
