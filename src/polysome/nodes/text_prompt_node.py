from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
import signal
import time
from functools import wraps
from polysome.nodes.jsonl_processing_node import JSONLProcessingNode
from polysome.nodes.node import ValidationResult, node_step_error_handler
from polysome.prompt_formatter import PromptFormatter
from polysome.engines.registry import get_engine
from polysome.utils.post_processing import extract_and_parse_json
from polysome.utils.jsonl_writer import IncrementalJsonlWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


def timeout_wrapper(timeout_seconds: float, error_message: str = "Operation timed out"):
    """
    Decorator that adds timeout functionality to a function.

    Args:
        timeout_seconds: Maximum time to wait for function completion
        error_message: Message to include in TimeoutError
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            class TimeoutException(Exception):
                pass

            def timeout_handler(signum, frame):
                raise TimeoutException(error_message)

            # Save the old handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)

            try:
                # Set the alarm
                signal.alarm(int(timeout_seconds))
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel the alarm
                return result
            except TimeoutException:
                raise TimeoutError(error_message)
            finally:
                # Restore the old handler
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


class TextPromptNode(JSONLProcessingNode):
    """LLM-based text processing node using the JSONLProcessingNode infrastructure."""

    # Default filenames
    DEFAULT_SYSTEM_PROMPT_FILE = "system_prompt.txt"
    DEFAULT_USER_PROMPT_FILE = "user_prompt.txt"
    DEFAULT_FEW_SHOT_LINES_FILE = "few_shot.jsonl"

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
        super().__init__(
            node_id,
            node_type,
            parent_wf_name,
            data_dir,
            output_dir,
            prompts_dir,
            params,
        )

        # LLM-specific parameters
        self.model_name = params.get("model_name")
        self.engine_name = params.get("inference_engine", "huggingface")
        self.engine_options = params.get("engine_options", {})
        self.generation_options = params.get("generation_options", {})
        self.template_context_map = params.get("template_context_map", {})
        self.parse_json = params.get("parse_json", False)
        self.batch_size = params.get("batch_size", 1)
        self.batch_timeout = params.get(
            "batch_timeout", 600.0
        )  # 10 minutes default batch timeout

        # Prompt configuration
        self.system_prompt_file = params.get(
            "system_prompt_file", self.DEFAULT_SYSTEM_PROMPT_FILE
        )
        self.user_prompt_file = params.get(
            "user_prompt_file", self.DEFAULT_USER_PROMPT_FILE
        )
        self.few_shot_lines_file = params.get(
            "few_shot_lines_file", self.DEFAULT_FEW_SHOT_LINES_FILE
        )
        self.num_few_shots = params.get("num_few_shots", 0)

        # Few-shot configuration
        self.few_shot_context_key = params.get("few_shot_context_key", "context")
        self.few_shot_assistant_key = params.get("few_shot_assistant_key", "assistant")
        self.few_shot_id_key = params.get("few_shot_id_key", "id")

        # Will be initialized in setup_processing
        self.prompt_formatter = None
        self.model = None

        logger.info(
            f"TextPromptNode '{self.node_id}' initialized with model '{self.model_name}'"
        )

    def get_required_parameters(self) -> List[str]:
        """Specify required parameters."""
        return ["model_name"]

    def get_parameter_type_specs(self) -> Dict[str, type | Tuple[type, ...]]:
        """Specify parameter types."""
        return {
            "model_name": str,
            "inference_engine": str,
            "engine_options": dict,
            "generation_options": dict,
            "template_context_map": dict,
            "parse_json": bool,
            "batch_size": int,
            "system_prompt_file": str,
            "user_prompt_file": str,
            "few_shot_lines_file": str,
            "num_few_shots": int,
            "few_shot_context_key": str,
            "few_shot_assistant_key": str,
            "few_shot_id_key": str,
        }

    def get_parameter_value_specs(self) -> Dict[str, Dict[str, Any]]:
        """Specify parameter value constraints."""
        return {
            "inference_engine": {
                "choices": ["huggingface", "llama_cpp", "vllm", "vllm_dp"]
            },
        }

    def _validate_custom_logic(self, result: ValidationResult) -> None:
        """Custom validation for TextPromptNode."""
        # Validate template_context_map has string keys
        template_context_map = self.params.get("template_context_map", {})
        if template_context_map and not all(
            isinstance(k, str) for k in template_context_map.keys()
        ):
            result.add_error(
                "invalid_template_context_map_keys",
                "template_context_map must have string keys",
                field="template_context_map",
                value=template_context_map,
            )

        # Validate prompt file paths exist (if we have access to the file system)
        if self._should_validate_filesystem():
            self._validate_prompt_files(result)

        # Validate few-shot configuration consistency
        self._validate_few_shot_config(result)

    def _validate_prompt_files(self, result: ValidationResult) -> None:
        """Validate that prompt files exist and are accessible."""
        prompt_dir = self.prompts_dir / self.name

        # Check if prompt directory exists
        if not prompt_dir.exists():
            result.add_warning(
                "missing_prompt_directory",
                f"Prompt directory does not exist: {prompt_dir}",
            )
            return

        # Validate system prompt file
        system_prompt_path = prompt_dir / self.system_prompt_file
        if not system_prompt_path.exists():
            result.add_error(
                "missing_system_prompt_file",
                f"System prompt file not found: {system_prompt_path}",
                field="system_prompt_file",
                value=self.system_prompt_file,
            )
        elif not system_prompt_path.is_file():
            result.add_error(
                "invalid_system_prompt_file",
                f"System prompt path is not a file: {system_prompt_path}",
                field="system_prompt_file",
                value=self.system_prompt_file,
            )

        # Validate user prompt file
        user_prompt_path = prompt_dir / self.user_prompt_file
        if not user_prompt_path.exists():
            result.add_error(
                "missing_user_prompt_file",
                f"User prompt file not found: {user_prompt_path}",
                field="user_prompt_file",
                value=self.user_prompt_file,
            )
        elif not user_prompt_path.is_file():
            result.add_error(
                "invalid_user_prompt_file",
                f"User prompt path is not a file: {user_prompt_path}",
                field="user_prompt_file",
                value=self.user_prompt_file,
            )

        # Validate few-shot file if needed
        if self.num_few_shots > 0:
            few_shot_path = prompt_dir / self.few_shot_lines_file
            if not few_shot_path.exists():
                result.add_error(
                    "missing_few_shot_file",
                    f"Few-shot examples file not found: {few_shot_path} (required when num_few_shots > 0)",
                    field="few_shot_lines_file",
                    value=self.few_shot_lines_file,
                )
            elif not few_shot_path.is_file():
                result.add_error(
                    "invalid_few_shot_file",
                    f"Few-shot examples path is not a file: {few_shot_path}",
                    field="few_shot_lines_file",
                    value=self.few_shot_lines_file,
                )

    def _validate_few_shot_config(self, result: ValidationResult) -> None:
        """Validate few-shot configuration consistency."""
        # Warn if num_few_shots is 0 but few_shot file is specified and different from default
        if (
            self.num_few_shots == 0
            and self.few_shot_lines_file != self.DEFAULT_FEW_SHOT_LINES_FILE
        ):
            result.add_warning(
                "unused_few_shot_file",
                "Few-shot file specified but num_few_shots is 0",
                field="few_shot_lines_file",
            )

        # Check that few-shot keys are non-empty strings
        few_shot_keys = [
            ("few_shot_context_key", self.few_shot_context_key),
            ("few_shot_assistant_key", self.few_shot_assistant_key),
            ("few_shot_id_key", self.few_shot_id_key),
        ]

        for key_name, key_value in few_shot_keys:
            if not key_value or not key_value.strip():
                result.add_error(
                    "empty_few_shot_key",
                    f"{key_name} cannot be empty",
                    field=key_name,
                    value=key_value,
                )

    def setup_processing(self) -> None:
        """Initialize LLM and prompt formatter before processing."""

        if not self.model_name:
            raise ValueError(f"Node '{self.node_id}': model_name is required")

        try:
            # Initialize prompt formatter (matching your existing PromptFormatter usage)
            prompt_dir = self.prompts_dir / self.name
            self.prompt_formatter = PromptFormatter(
                system_prompt_path=prompt_dir / self.system_prompt_file,
                user_prompt_template_path=prompt_dir / self.user_prompt_file,
                few_shot_examples_path=prompt_dir / self.few_shot_lines_file,
                num_few_shots=self.num_few_shots,
                few_shot_context_key=self.few_shot_context_key,
                few_shot_assistant_key=self.few_shot_assistant_key,
                few_shot_id_key=self.few_shot_id_key,
            )

            # Try to acquire shared engine first, fall back to creating new one
            self.model = self.acquire_shared_engine()

            if self.model is None:
                # Fall back to creating new engine (non-shared)
                logger.info(
                    f"Node '{self.node_id}': Creating non-shared engine for model '{self.model_name}'"
                )
                try:
                    self.model = get_engine(
                        engine_name=self.engine_name,
                        model_name=self.model_name,
                        **self.engine_options,
                    )
                except Exception as e:
                    logger.error(
                        f"Node '{self.node_id}': Failed to create non-shared engine: {e}"
                    )
                    # Make sure we clean up any partial state
                    self.cleanup_processing()
                    raise RuntimeError(
                        f"Failed to create engine for model '{self.model_name}': {e}"
                    ) from e

            logger.info(
                f"Node '{self.node_id}': LLM setup complete - {self.model_name}"
            )

        except Exception as e:
            logger.error(f"Node '{self.node_id}': Failed to setup processing: {e}")
            # Ensure cleanup on any setup failure
            self.cleanup_processing()
            raise

    def cleanup_processing(self) -> None:
        """Clean up resources after processing."""
        # If using shared engines, the base class will handle release
        # If using non-shared engines, we need to unload manually
        if self.model is not None and not self.use_shared_engines:
            try:
                logger.info(
                    f"Node '{self.node_id}': Unloading non-shared model to free memory"
                )
                self.model.unload_model()
            except Exception as e:
                logger.error(f"Node '{self.node_id}': Error during model unload: {e}")
            finally:
                self.model = None
        elif self.model is not None:
            logger.debug(
                f"Node '{self.node_id}': Model cleanup will be handled by shared engine pool"
            )
            self.model = None

        # Clear prompt formatter as well
        self.prompt_formatter = None

    def process_item(self, key: str, row_data: Dict[str, Any]) -> Any:
        """Process item using LLM."""
        logger.debug(f"Node '{self.node_id}': Processing item with key: {key}")

        # Prepare template context (matching your existing logic)
        if self.template_context_map:
            template_context = {}
            for template_var, data_key in self.template_context_map.items():
                if data_key in row_data:
                    template_context[template_var] = row_data[data_key]
                else:
                    logger.warning(
                        f"Node '{self.node_id}', item '{key}': Data key '{data_key}' for template variable "
                        f"'{template_var}' not found in row_data. Variable will be missing or empty in template."
                    )
                    template_context[template_var] = ""
        else:
            # No map: pass all row_data attributes directly
            template_context = row_data.copy()

        assert self.prompt_formatter is not None and self.model is not None, (
            f"Node '{self.node_id}': Prompt formatter and model must be initialized before processing items."
        )

        # Generate prompt and get LLM response
        messages = self.prompt_formatter.create_messages(template_context)
        output = self.model.generate_text(messages, **self.generation_options)

        # Parse JSON if requested
        if self.parse_json:
            parsed_output = extract_and_parse_json(output)
            # provide text as is if parsing fails
            if parsed_output is not None:
                output = parsed_output

        return output

    def _execute_processing(self, data_to_process: Dict[str, Any], items_count: int):
        """Execute the main processing loop with optional batching."""
        if self.batch_size <= 1 or not self.model.supports_native_batching():
            # Use default single-item processing
            if self.batch_size > 1 and not self.model.supports_native_batching():
                logger.info(
                    f"Node '{self.node_id}': Engine '{self.engine_name}' does not support native batching. "
                    f"Using single-item processing."
                )
            super()._execute_processing(data_to_process, items_count)
        else:
            # Use batch processing
            self._execute_batch_processing(data_to_process, items_count)

    @node_step_error_handler(failure_status="failed_batch_processing_execution")
    def _execute_batch_processing(
        self, data_to_process: Dict[str, Any], items_count: int
    ):
        """Execute processing using batching."""

        try:
            logger.info(f"Node '{self.node_id}': Starting batch processing setup")
            
            # Ensure output directory exists
            self.output_full_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Node '{self.node_id}': Output directory created/verified")

            # Determine file mode based on resume setting
            file_mode = "a" if self.resume else "w"
            logger.info(f"Node '{self.node_id}': Opening output file in mode '{file_mode}' (resume={self.resume})")

            try:
                with IncrementalJsonlWriter(self.output_full_path, mode=file_mode) as writer:
                    logger.info(f"Node '{self.node_id}': JSONL writer context entered successfully")
                    logger.info(
                        f"Node '{self.node_id}': Processing {items_count} items in batches of {self.batch_size} -> {self.output_full_path}"
                    )

                    # Convert data to list for batching
                    items = list(data_to_process.items())

                    # Process in batches
                    for batch_start in tqdm(
                        range(0, len(items), self.batch_size),
                        desc=f"Processing {self.node_id} (batched)",
                        total=(len(items) + self.batch_size - 1) // self.batch_size,
                    ):
                        batch_end = min(batch_start + self.batch_size, len(items))
                        batch_items = items[batch_start:batch_end]

                        # Prepare batch data
                        batch_keys = []
                        batch_messages = []
                        batch_row_data = []

                        for key, row_data in batch_items:
                            # Prepare template context for this item
                            if self.template_context_map:
                                template_context = {}
                                for (
                                    template_var,
                                    data_key,
                                ) in self.template_context_map.items():
                                    if data_key in row_data:
                                        template_context[template_var] = row_data[data_key]
                                    else:
                                        logger.warning(
                                            f"Node '{self.node_id}', item '{key}': Data key '{data_key}' for template variable "
                                            f"'{template_var}' not found in row_data. Variable will be missing or empty in template."
                                        )
                                        template_context[template_var] = ""
                            else:
                                # No map: pass all row_data attributes directly
                                template_context = row_data.copy()

                            # Generate messages for this item
                            messages = self.prompt_formatter.create_messages(
                                template_context
                            )

                            batch_keys.append(key)
                            batch_messages.append(messages)
                            batch_row_data.append(row_data)

                        # Process the entire batch with timeout
                        try:
                            # Create a timeout wrapper for the batch processing
                            @timeout_wrapper(
                                self.batch_timeout,
                                f"Batch processing timed out after {self.batch_timeout} seconds",
                            )
                            def process_batch_with_timeout():
                                return self.model.generate_text_batch(
                                    batch_messages, **self.generation_options
                                )

                            logger.debug(
                                f"Node '{self.node_id}': Starting batch processing with timeout {self.batch_timeout}s"
                            )
                            batch_outputs = process_batch_with_timeout()

                            # Process batch results
                            for i, (key, row_data, output) in enumerate(
                                zip(batch_keys, batch_row_data, batch_outputs)
                            ):
                                try:
                                    # Parse JSON if requested
                                    if self.parse_json:
                                        parsed_output = extract_and_parse_json(output)
                                        if parsed_output is not None:
                                            output = parsed_output

                                    # Build output record
                                    output_record = {
                                        self.primary_key: str(key),
                                        self.output_data_attribute: output,
                                    }

                                    # Include original data attributes
                                    for orig_key, orig_value in row_data.items():
                                        if orig_key not in output_record:
                                            output_record[orig_key] = orig_value

                                    writer.write_row(output_record)

                                except Exception as e:
                                    logger.error(
                                        f"Node '{self.node_id}': Error processing batch item {key}: {e}"
                                    )
                                    self.errors.append(f"Item {key}: {e}")

                        except TimeoutError as e:
                            logger.error(
                                f"Node '{self.node_id}': Batch processing timed out starting at {batch_start}: {e}"
                            )
                            # Add timeout errors for all items in the failed batch
                            for key in batch_keys:
                                self.errors.append(
                                    f"Item {key}: Batch processing timed out: {e}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Node '{self.node_id}': Error processing batch starting at {batch_start}: {e}"
                            )
                            # Add errors for all items in the failed batch
                            for key in batch_keys:
                                self.errors.append(
                                    f"Item {key}: Batch processing failed: {e}"
                                )
            
            except (IOError, OSError, PermissionError) as e:
                logger.error(f"Node '{self.node_id}': File access error opening JSONL writer: {e}")
                raise RuntimeError(f"Failed to open output file {self.output_full_path}: {e}") from e

        except IOError as e:
            logger.error(
                f"Node '{self.node_id}': I/O error during batch processing: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Node '{self.node_id}': Unexpected error during batch processing: {e}"
            )
            raise
