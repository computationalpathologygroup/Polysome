from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Set, List, Callable, Tuple
import json
import logging
from tqdm import tqdm
from dataclasses import dataclass
from polysome.utils.jsonl_writer import IncrementalJsonlWriter
from polysome.utils.data_loader import DataFileLoader
from polysome.nodes.node import (
    BaseNode,
    node_step_error_handler,
    processing_exception_handler,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class JSONLProcessingNode(BaseNode, ABC):
    """
    Abstract base class for nodes that process JSONL data item-by-item.

    This class provides:
    - JSONL file loading and saving infrastructure
    - Resume functionality
    - Error handling for individual items
    - Template method pattern for processing logic

    Subclasses only need to implement process_item() method.
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
        super().__init__(
            node_id, node_type, parent_wf_name, data_dir, output_dir, prompts_dir, params
        )

        # Processing-specific parameters
        self.resume = params.get("resume", False)
        self.output_data_attribute = params.get("output_data_attribute", "output")
        
        # Engine sharing parameters
        self.use_shared_engines = params.get("use_shared_engines", True)
        self.engine_name = params.get("inference_engine")
        self.model_name = params.get("model_name")
        self.engine_options = params.get("engine_options", {})
        self.engine_timeout = params.get("engine_timeout", 300.0)  # 5 minutes default timeout

        # Will be initialized during run
        self.data_loader: Optional[DataFileLoader] = None
        self.shared_engine = None  # For shared engine instances

        logger.info(f"JSONLProcessingNode '{self.node_id}' initialized.")
        logger.info(f"  Resume enabled: {self.resume}")
        logger.info(f"  Output attribute: {self.output_data_attribute}")
        if self.model_name:
            logger.info(f"  Engine sharing enabled: {self.use_shared_engines}")
            logger.info(f"  Model: {self.model_name} (engine: {self.engine_name})")

    @abstractmethod
    def process_item(self, key: str, row_data: Dict[str, Any]) -> Any:
        """
        Process a single data item.

        Args:
            key: The primary key value for this item
            row_data: Dictionary containing all data for this item

        Returns:
            The processed result (can be any type - string, dict, etc.)

        Raises:
            Exception: Any processing errors will be caught and logged by the framework
        """
        pass

    def setup_processing(self) -> None:
        """
        Hook for subclasses to perform additional setup before processing.
        Called after input resolution but before data loading.
        Override if needed (e.g., to initialize models, load templates, etc.)
        """
        pass

    def cleanup_processing(self) -> None:
        """
        Hook for subclasses to perform cleanup after processing.
        Called at the end of run() regardless of success/failure.
        Override if needed (e.g., to close model connections, etc.)
        """
        pass
    
    def acquire_shared_engine(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Acquire a shared engine from the engine pool if sharing is enabled.
        
        Args:
            timeout: Maximum time to wait for engine acquisition (uses node default if None)
        
        Returns:
            Engine instance if sharing is enabled and model_name is set, None otherwise
        """
        if not self.use_shared_engines or not self.model_name or not self.engine_name:
            return None
            
        if timeout is None:
            timeout = self.engine_timeout
            
        try:
            from polysome.engines.engine_pool import get_engine_pool
            engine_pool = get_engine_pool()
            
            engine = engine_pool.acquire_engine(
                engine_name=self.engine_name,
                model_name=self.model_name,
                engine_options=self.engine_options,
                node_id=self.node_id,
                timeout=timeout
            )
            
            self.shared_engine = engine
            logger.info(f"Node '{self.node_id}': Acquired shared engine '{self.engine_name}' for model '{self.model_name}'")
            return engine
            
        except Exception as e:
            logger.error(f"Node '{self.node_id}': Failed to acquire shared engine: {e}")
            return None
    
    def release_shared_engine(self, timeout: Optional[float] = None) -> None:
        """
        Release the shared engine back to the pool if sharing is enabled.
        
        Args:
            timeout: Maximum time to wait for engine release (uses node default if None)
        """
        if not self.use_shared_engines or not self.shared_engine:
            return
            
        if timeout is None:
            timeout = self.engine_timeout
            
        try:
            from polysome.engines.engine_pool import get_engine_pool
            engine_pool = get_engine_pool()
            
            engine_pool.release_engine(
                engine_name=self.engine_name,
                model_name=self.model_name,
                engine_options=self.engine_options,
                node_id=self.node_id,
                timeout=timeout
            )
            
            logger.info(f"Node '{self.node_id}': Released shared engine '{self.engine_name}' for model '{self.model_name}'")
            self.shared_engine = None
            
        except Exception as e:
            logger.error(f"Node '{self.node_id}': Failed to release shared engine: {e}")

    @node_step_error_handler(failure_status="failed_resolve_input")
    def _resolve_input(self, input_data: Dict[str, Any] | None = None):
        """Resolve input data path and primary key from dependencies or params."""
        if input_data:
            if len(input_data) > 1:
                logger.warning(
                    f"Node '{self.node_id}': Multiple dependencies found, using first."
                )

            dep_id, dep_output = next(iter(input_data.items()))

            # Set input path from dependency
            resolved_path_str = dep_output.get("output_path")
            if not resolved_path_str:
                raise ValueError(f"Dependency '{dep_id}' missing 'output_path'.")
            self.input_data_path = Path(resolved_path_str)

            # Inherit primary key if not set
            if not self.primary_key and dep_output.get("primary_key"):
                self.primary_key = dep_output["primary_key"]
                logger.info(
                    f"Node '{self.node_id}': Inherited primary_key '{self.primary_key}' from '{dep_id}'."
                )

            # Inherit data attribute hint if not set
            if not self.data_attribute and dep_output.get("output_attribute"):
                self.data_attribute = dep_output["output_attribute"]
                logger.info(
                    f"Node '{self.node_id}': Inherited data_attribute '{self.data_attribute}' from '{dep_id}'."
                )

            logger.info(
                f"Node '{self.node_id}': Input from dependency '{dep_id}' -> {self.input_data_path}"
            )

        elif not self.input_data_path:
            raise ValueError(
                f"Node '{self.node_id}': No input_data_path configured and no dependencies."
            )

        # Validate required fields
        if not self.input_data_path:
            raise ValueError(
                f"Node '{self.node_id}': input_data_path could not be resolved."
            )
        if not self.primary_key:
            raise ValueError(
                f"Node '{self.node_id}': primary_key could not be resolved."
            )

    @node_step_error_handler(failure_status="failed_init_data_loader")
    def _initialize_data_loader(self):
        """Initialize the data loader."""
        if not self.input_data_path or not self.primary_key:
            raise RuntimeError(
                f"Node '{self.node_id}': input_data_path and primary_key must be set."
            )

        logger.info(
            f"Node '{self.node_id}': Initializing data loader for {self.input_data_path}"
        )
        self.data_loader = DataFileLoader(
            input_data_path=self.input_data_path,
            primary_key=self.primary_key,
        )

    def _load_processed_ids(self) -> Set[str]:
        """Load already processed item IDs for resume functionality."""
        processed_ids = set()
        if not self.output_full_path.exists():
            return processed_ids

        logger.info(f"Node '{self.node_id}': Loading processed IDs for resume...")
        try:
            with open(self.output_full_path, "r", encoding="utf-8") as f:
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

    @node_step_error_handler(failure_status="failed_load_data")
    def _load_and_filter_data(self) -> tuple[Optional[Dict[str, Any]], int]:
        """Load data and filter out already processed items if resuming."""
        if not self.data_loader:
            raise RuntimeError(f"Node '{self.node_id}': Data loader not initialized.")

        # Load all data
        logger.info(f"Node '{self.node_id}': Loading data from {self.input_data_path}")
        all_data = self.data_loader.load_input_data()
        total_items = len(all_data)

        if not all_data:
            logger.warning(
                f"Node '{self.node_id}': No data loaded from {self.input_data_path}"
            )
            return None, 0

        logger.info(f"Node '{self.node_id}': Loaded {total_items} items")

        # Apply resume filtering if enabled
        logger.debug(f"Node '{self.node_id}': Checking resume flag: {self.resume}")
        if self.resume:
            logger.info(f"Node '{self.node_id}': Resume enabled, loading processed IDs...")
            processed_ids = self._load_processed_ids()
            logger.info(f"Node '{self.node_id}': Starting data filtering with {len(processed_ids)} processed IDs...")
            data_to_process = {
                k: v for k, v in all_data.items() if str(k) not in processed_ids
            }
            logger.info(f"Node '{self.node_id}': Data filtering completed")
            skipped = total_items - len(data_to_process)
            if skipped > 0:
                logger.info(
                    f"Node '{self.node_id}': Resume - skipping {skipped} already processed items"
                )
        else:
            logger.info(f"Node '{self.node_id}': Resume disabled, using all data")
            data_to_process = all_data

        logger.debug(f"Node '{self.node_id}': Calculating items to process...")
        items_to_process = len(data_to_process)
        if items_to_process == 0:
            logger.info(f"Node '{self.node_id}': No items to process")
            return None, 0

        logger.info(
            f"Node '{self.node_id}': {items_to_process} items selected for processing"
        )
        return data_to_process, items_to_process

    @processing_exception_handler(error_list_attr="errors", key_arg_index=1)
    def _process_item_wrapper(
        self, key: str, row_data: Dict[str, Any]
    ) -> Optional[Any]:
        """Wrapper around process_item that handles exceptions."""
        return self.process_item(key, row_data)

    @node_step_error_handler(failure_status="failed_processing_execution")
    def _execute_processing(self, data_to_process: Dict[str, Any], items_count: int):
        """Execute the main processing loop."""
        try:
            # Ensure output directory exists
            self.output_full_path.parent.mkdir(parents=True, exist_ok=True)

            with IncrementalJsonlWriter(self.output_full_path) as writer:
                logger.info(
                    f"Node '{self.node_id}': Processing {items_count} items -> {self.output_full_path}"
                )

                for key, row_data in tqdm(
                    data_to_process.items(),
                    desc=f"Processing {self.node_id}",
                    total=items_count,
                ):
                    # Process the item (exceptions handled by decorator)
                    processed_result = self._process_item_wrapper(key, row_data)

                    if processed_result is not None:
                        # Build output record
                        output_record = {
                            self.primary_key: str(key),
                            self.output_data_attribute: processed_result,
                        }

                        # Include original data attributes
                        for orig_key, orig_value in row_data.items():
                            if orig_key not in output_record:
                                output_record[orig_key] = orig_value

                        writer.write_row(output_record)

        except IOError as e:
            logger.error(f"Node '{self.node_id}': I/O error during processing: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Node '{self.node_id}': Unexpected error during processing: {e}"
            )
            raise

    def _prepare_output_info(self, status: str, error_count: int) -> Dict[str, Any]:
        """Prepare the output info dictionary."""
        return {
            "output_path": str(self.output_full_path),
            "output_attribute": self.output_data_attribute,
            "primary_key": self.primary_key,
            "status": status,
            "errors_count": error_count,
        }

    def _execute_pipeline(self, steps: List[tuple[str, Callable]]) -> Any:
        """
        Execute a pipeline of steps, automatically handling status checks.

        Args:
            steps: List of (step_name, callable) tuples

        Returns:
            The result of the last step, or None if any step failed
        """
        result = None

        for step_name, step_func in steps:
            if self.status != "running":
                logger.debug(
                    f"Node '{self.node_id}': Skipping '{step_name}' due to status: {self.status}"
                )
                break

            try:
                logger.debug(
                    f"Node '{self.node_id}': Executing pipeline step: {step_name}"
                )
                result = step_func()

                # Special handling for setup_processing which doesn't use decorators
                if step_name == "setup_processing" and self.status == "running":
                    continue  # Setup successful, continue to next step

            except Exception as e:
                if step_name == "setup_processing":
                    logger.error(f"Node '{self.node_id}': Setup failed: {e}")
                    self.status = "failed_setup"
                # Other exceptions should be handled by the step's decorators
                break

        return result

    def run(self, input_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Main execution method using template method pattern with pipeline execution."""
        logger.info(f"--- Starting JSONLProcessingNode '{self.node_id}' ---")

        # Reset state
        self.errors = []
        self.status = "running"
        items_processed = 0

        try:
            # Define the processing pipeline
            pipeline_result = self._execute_pipeline(
                [
                    ("resolve_input", lambda: self._resolve_input(input_data)),
                    ("initialize_data_loader", self._initialize_data_loader),
                    ("setup_processing", self.setup_processing),
                    ("load_and_filter_data", self._load_and_filter_data),
                ]
            )

            # Handle data processing if pipeline succeeded
            if self.status == "running":
                data_to_process, items_count = (
                    pipeline_result if pipeline_result else (None, 0)
                )

                if data_to_process and items_count > 0:
                    items_processed = items_count
                    self._execute_processing(data_to_process, items_count)

                    # Determine final status
                    if self.status == "running":
                        if self.errors:
                            self.status = "completed_with_errors"
                        else:
                            self.status = "completed_successfully"
                else:
                    self.status = "completed_no_new_items"

        finally:
            # Always run cleanup
            try:
                self.cleanup_processing()
            except Exception as e:
                logger.warning(f"Node '{self.node_id}': Cleanup error (ignored): {e}")
            
            # Release shared engine (safe with deferred cleanup enabled by workflow)
            self.release_shared_engine()

        # Final reporting
        error_count = len(self.errors)
        logger.info(
            f"Node '{self.node_id}': Completed - {items_processed} items processed, "
            f"{error_count} errors, status: {self.status}"
        )

        output_info = self._prepare_output_info(self.status, error_count)
        logger.info(f"--- Finished JSONLProcessingNode '{self.node_id}' ---")
        return output_info
