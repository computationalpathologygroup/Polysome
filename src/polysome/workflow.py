import json
from pathlib import Path
import logging
from typing import Dict, List, Any, Union, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Import Node classes and the map
from polysome.nodes.node import BaseNode
from polysome.nodes.node_registry import NODE_TYPE_MAP
from polysome.execution_optimizer import ExecutionOptimizer
from polysome.utils.tree_utils import generate_execution_tree_ascii

logger = logging.getLogger(__name__)


# =====================================================================
# VALIDATION DATA STRUCTURES
# =====================================================================


@dataclass
class NodeValidationInfo:
    """Holds the validation result for a single node."""

    node_id: str
    node_type: str
    is_valid: bool = False
    has_warnings: bool = False
    summary: str = ""
    detailed_report: str = ""
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the original dictionary format for compatibility."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "is_valid": self.is_valid,
            "has_warnings": self.has_warnings,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "summary": self.summary,
            "detailed_report": self.detailed_report,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class WorkflowValidationReport:
    """Manages the overall validation report for the entire workflow."""

    workflow_name: str
    total_nodes: int
    nodes_with_errors: List[str] = field(default_factory=list)
    nodes_with_warnings: List[str] = field(default_factory=list)
    validation_details: Dict[str, NodeValidationInfo] = field(default_factory=dict)
    total_errors: int = 0
    total_warnings: int = 0

    def add_node_result(self, node_info: NodeValidationInfo) -> None:
        """Adds a node's validation result and updates summary counts."""
        self.validation_details[node_info.node_id] = node_info
        self.total_errors += node_info.error_count
        self.total_warnings += node_info.warning_count

        if not node_info.is_valid:
            self.nodes_with_errors.append(node_info.node_id)
        if node_info.has_warnings:
            self.nodes_with_warnings.append(node_info.node_id)

    @property
    def valid_nodes_count(self) -> int:
        return self.total_nodes - len(self.nodes_with_errors)

    @property
    def is_overall_valid(self) -> bool:
        return self.total_errors == 0

    def to_dict(self) -> Dict[str, Any]:
        """Converts the report to the original dictionary format for compatibility."""
        return {
            "workflow_name": self.workflow_name,
            "total_nodes": self.total_nodes,
            "nodes_with_errors": self.nodes_with_errors,
            "nodes_with_warnings": self.nodes_with_warnings,
            "validation_details": {
                node_id: info.to_dict()
                for node_id, info in self.validation_details.items()
            },
            "summary": {
                "total_errors": self.total_errors,
                "total_warnings": self.total_warnings,
                "valid_nodes": self.valid_nodes_count,
                "invalid_nodes": len(self.nodes_with_errors),
            },
        }


class MockOutputManager:
    """Manages mock dependency outputs for validation purposes."""

    def __init__(self, workflow_instance: "Workflow"):
        self.workflow = workflow_instance
        self.mock_outputs: Dict[str, Dict[str, Any]] = {}

    def get_mock_input_for_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Prepare mock input data for a node from its dependencies."""
        input_data = {}
        node_dependencies = self.workflow.dependencies.get(node_id, [])

        for dep_id in node_dependencies:
            if dep_id in self.mock_outputs:
                input_data[dep_id] = self.mock_outputs[dep_id]
            else:
                # Create and cache mock output for dependency
                mock_output = self.workflow._create_mock_dependency_output(dep_id)
                self.mock_outputs[dep_id] = mock_output
                input_data[dep_id] = mock_output

        return input_data if input_data else None

    def add_mock_output(self, node_id: str) -> None:
        """Create and store mock output for a node."""
        if node_id not in self.mock_outputs:
            self.mock_outputs[node_id] = self.workflow._create_mock_dependency_output(
                node_id
            )


class WorkflowValidationError(Exception):
    """Raised when workflow validation fails."""

    pass


class Workflow:
    """
    Manages and executes a workflow defined in a JSON configuration file.
    The workflow represents a Directed Acyclic Graph (DAG) of processing nodes.
    """

    def __init__(
        self, config_path: Union[str, Path], optimize_for_engines: bool = True
    ):
        """
        Initializes the Workflow runner.

        Args:
            config_path: Path to the workflow JSON configuration file.
            optimize_for_engines: Whether to optimize execution order for engine sharing.
        """
        self.config_path = Path(config_path)
        self.workflow_name = "unnamed_workflow"
        self.nodes_config: Dict[str, Dict] = {}
        self.dependencies: Dict[str, List[str]] = defaultdict(
            list
        )  # node_id -> [dependency_id, ...]
        self.dependents: Dict[str, List[str]] = defaultdict(
            list
        )  # node_id -> [dependent_id, ...]
        self.node_instances: Dict[str, BaseNode] = {}
        self.node_outputs: Dict[
            str, Dict[str, Any]
        ] = {}  # node_id -> output_info dict from node.run()
        self.execution_order: List[str] = []
        self.optimize_for_engines = optimize_for_engines

        self.config = self._load_and_parse_config()
        self.data_dir = Path(self.config["data_dir"])
        self.output_dir = Path(self.config["output_dir"])
        self.prompts_dir = Path(self.config["prompts_dir"])
        self.log_dir = Path(self.config.get("log_dir", self.output_dir / "logs"))

        # Check for engine optimization setting in config
        workflow_config = self.config.get("workflow_settings", {})
        if "optimize_for_engines" in workflow_config:
            self.optimize_for_engines = workflow_config["optimize_for_engines"]
            logger.info(
                f"Engine optimization setting from config: {self.optimize_for_engines}"
            )

        self._build_dag()
        self._validate_dag()
        self._determine_execution_order(self.optimize_for_engines)

    def _process_engine_cleanup_marker(self, item: str):
        try:
            from polysome.engines.engine_pool import get_engine_pool
            engine_pool = get_engine_pool()
            parts = item.replace("__ENGINE_CLEANUP__", "").replace("__", "").split("_TO_")
            if len(parts) == 2:
                from_engine_key = parts[0]
                to_engine_key = parts[1]
                
                logger.info(f"Processing engine cleanup marker: {from_engine_key} → {to_engine_key}")
                
                # Get current engine stats to see what's actually loaded
                engine_stats = engine_pool.get_engine_stats()
                if not engine_stats:
                    logger.debug("No engines currently loaded, skipping cleanup")
                    return
                
                # Find engines to keep (those that match the target pattern)
                engines_to_keep = []
                if to_engine_key != "_no_engine_":
                    for full_key in engine_stats.keys():
                        if full_key.startswith(to_engine_key):
                            engines_to_keep.append(full_key)
                
                # Clean up engines that don't match the target
                engines_to_cleanup = []
                for full_key in engine_stats.keys():
                    if to_engine_key == "_no_engine_" or not full_key.startswith(to_engine_key):
                        engines_to_cleanup.append(full_key)
                
                if engines_to_cleanup:
                    logger.info(f"Cleaning up engines: {engines_to_cleanup}")
                    logger.info(f"Keeping engines: {engines_to_keep}")
                    # Use the first engine to cleanup as the from_engine, any target as to_engine
                    target_key = engines_to_keep[0] if engines_to_keep else "_no_engine_"
                    engine_pool.force_cleanup_between_engines(
                        from_engine_key=engines_to_cleanup[0], 
                        to_engine_key=target_key,
                        exclude_keys=engines_to_keep
                    )
                else:
                    logger.debug("No engines need cleanup")
                        
        except Exception as e:
            logger.warning(f"Error processing engine cleanup marker {item}: {e}")
            pass

    def _load_and_parse_config(self) -> Dict:
        """Loads and performs basic validation on the workflow JSON."""
        logger.info(f"Loading workflow configuration from: {self.config_path}")
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.error(f"Workflow configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.config_path}: {e}")
            raise

        self.workflow_name = config.get("name", self.config_path.stem)
        logger.info(f"Loaded workflow: '{self.workflow_name}'")

        if "nodes" not in config or not isinstance(config["nodes"], list):
            raise ValueError("Workflow config must contain a 'nodes' list.")

        for node_def in config["nodes"]:
            if not all(k in node_def for k in ["id", "type", "params", "dependencies"]):
                raise ValueError(
                    f"Node definition missing required keys (id, type, params, dependencies): {node_def}"
                )

            node_id = node_def["id"]
            if node_id in self.nodes_config:
                raise ValueError(f"Duplicate node ID found in config: {node_id}")

            if node_def["type"] not in NODE_TYPE_MAP:
                raise ValueError(
                    f"Unknown node type '{node_def['type']}' for node ID '{node_id}'. Available types: {list(NODE_TYPE_MAP.keys())}"
                )

            self.nodes_config[node_id] = node_def
            logger.debug(f"Parsed node config for ID: {node_id}")
        return config

    def _build_dag(self):
        """Builds dependency and dependent tracking structures."""
        all_node_ids = set(self.nodes_config.keys())
        for node_id, node_def in self.nodes_config.items():
            deps = node_def.get("dependencies", [])
            if not isinstance(deps, list):
                raise ValueError(f"Node '{node_id}': 'dependencies' must be a list.")

            for dep_id in deps:
                if dep_id not in all_node_ids:
                    raise ValueError(
                        f"Node '{node_id}' declares dependency on unknown node ID: '{dep_id}'"
                    )
                self.dependencies[node_id].append(dep_id)
                self.dependents[dep_id].append(node_id)
            # Initialize dependents entry even for nodes with no dependents
            if node_id not in self.dependents:
                self.dependents[node_id] = []
        logger.debug("Built DAG dependency structure.")

    def _determine_execution_order(self, optimize_for_engines: bool = True):
        """
        Determines a valid execution order using the ExecutionOptimizer.

        Args:
            optimize_for_engines: If True, optimize order to minimize model loading/unloading
        """
        optimizer = ExecutionOptimizer(
            self.nodes_config, self.dependencies, self.dependents
        )
        result = optimizer.determine_execution_order(optimize_for_engines)
        
        self.execution_order = result.execution_order
        self.cleanup_points = result.cleanup_points

        logger.info(f"Determined execution order: {self.execution_order}")

        if optimize_for_engines:
            self._log_engine_optimization_details(
                result.engine_groups, result.basic_order, result.execution_order
            )

    def _log_engine_optimization_details(
        self,
        engine_groups: Dict[str, List[str]],
        basic_order: List[str],
        optimized_order: List[str],
    ) -> None:
        """Log detailed information about multi-level engine optimization."""
        logger.info("=== Multi-Level Engine Optimization Details ===")

        # Log original engine groups
        logger.info("Original engine groups:")
        for engine_key, nodes in engine_groups.items():
            if engine_key == "_no_engine_":
                logger.info(f"  Non-engine nodes: {nodes}")
            else:
                # Parse engine key to show readable format
                parts = engine_key.split("::")
                if len(parts) >= 2:
                    engine_name = parts[0]
                    model_name = parts[1]
                    # Show engine options for better visibility
                    if len(parts) >= 3:
                        try:
                            import json
                            options = json.loads(parts[2])
                            key_options = {k: v for k, v in options.items() if k in ['max_model_len', 'data_parallel_size', 'gpu_memory_utilization']}
                            if key_options:
                                logger.info(f"  {engine_name} '{model_name}' {key_options}: {nodes}")
                            else:
                                logger.info(f"  {engine_name} '{model_name}': {nodes}")
                        except Exception as e:
                            logger.info(f"  {engine_name} '{model_name}': {nodes} (Error: {e})")
                    else:
                        logger.info(f"  {engine_name} '{model_name}': {nodes}")
                else:
                    logger.info(f"  {engine_key}: {nodes}")

        # Count cleanup points
        cleanup_count = len([item for item in optimized_order if item.startswith("__ENGINE_CLEANUP__")])
        if cleanup_count > 0:
            logger.info(f"Optimization inserted {cleanup_count} mandatory cleanup points")
            
            # Show cleanup transitions
            for i, item in enumerate(optimized_order):
                if item.startswith("__ENGINE_CLEANUP__"):
                    parts = item.replace("__ENGINE_CLEANUP__", "").replace("__", "").split("_TO_")
                    if len(parts) == 2:
                        logger.info(f"  Cleanup point {i}: {parts[0]} → {parts[1]}")
        else:
            logger.info("No engine cleanup points needed (single engine configuration)")
        
        # Compare orders
        if hasattr(self, 'cleanup_points') and self.cleanup_points:
            logger.info(f"Execution will include {len(self.cleanup_points)} forced cleanup operations")
        
        # Generate and log the ASCII execution tree
        logger.info("")
        ascii_tree = generate_execution_tree_ascii(optimized_order, self.nodes_config, self.dependencies)
        logger.info(ascii_tree)
        logger.info("")
        
        logger.info("=== End Optimization Details ===")



    def _group_nodes_by_engine(self) -> Dict[str, List[str]]:
        """
        Group nodes by their engine configuration.

        Returns:
            Dictionary mapping engine keys to lists of node IDs
        """
        engine_groups = defaultdict(list)

        for node_id, node_config in self.nodes_config.items():
            params = node_config.get("params", {})
            model_name = params.get("model_name")

            if model_name:
                engine_name = params.get("inference_engine", "huggingface")
                engine_options = params.get("engine_options", {})
                sorted_options = json.dumps(engine_options, sort_keys=True)
                engine_key = f"{engine_name}::{model_name}::{sorted_options}"
                engine_groups[engine_key].append(node_id)
            else:
                engine_groups["_no_engine_"].append(node_id)

        return engine_groups

    def _create_mock_dependency_output(self, node_id: str) -> Dict[str, Any]:
        """
        Create a mock output structure for validation purposes.
        This simulates what a dependency node would output.
        """
        node_config = self.nodes_config[node_id]

        # Create basic mock output structure
        mock_output = {
            "output_path": str(self.output_dir / f"{node_id}_output.jsonl"),
            "output_attribute": "output",  # Default attribute name
            "primary_key": "id",  # Default primary key
            "status": "completed_successfully",
            "errors_count": 0,
        }

        # Try to infer better values from node params if available
        params = node_config.get("params", {})
        if "output_data_attribute" in params:
            mock_output["output_attribute"] = params["output_data_attribute"]
        if "primary_key" in params:
            mock_output["primary_key"] = params["primary_key"]

        return mock_output

    def _instantiate_node(self, node_id: str) -> BaseNode:
        """Instantiate a single node."""
        node_config = self.nodes_config[node_id]
        node_type = node_config["type"]
        node_params = node_config["params"]
        node_class = NODE_TYPE_MAP[node_type]

        return node_class(
            node_id=node_id,
            node_type=node_type,
            parent_wf_name=self.workflow_name,
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            prompts_dir=self.prompts_dir,
            params=node_params,
        )

    # =====================================================================
    # VALIDATION METHODS
    # =====================================================================

    def _validate_dag(self):
        """Checks for cycles in the DAG using Kahn's algorithm (basis)."""
        in_degree = {
            node_id: len(self.dependencies[node_id]) for node_id in self.nodes_config
        }
        queue = deque(
            [node_id for node_id in self.nodes_config if in_degree[node_id] == 0]
        )
        count = 0

        while queue:
            u = queue.popleft()
            count += 1

            for v in self.dependents[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if count != len(self.nodes_config):
            # More sophisticated cycle detection could pinpoint the cycle
            logger.error("Workflow DAG contains a cycle!")
            raise ValueError("Workflow configuration results in a cyclic dependency.")
        logger.info("Workflow DAG validation successful (no cycles detected).")

    def _validate_execution_order(self, order: List[str]) -> bool:
        """
        Validate that an execution order respects all dependencies.
        Now handles cleanup markers in the execution order.

        Args:
            order: Proposed execution order (may include cleanup markers)

        Returns:
            True if order is valid, False otherwise
        """
        # Filter out cleanup markers for validation
        node_order = [item for item in order if not item.startswith("__ENGINE_CLEANUP__")]
        
        if len(node_order) != len(self.nodes_config):
            logger.error(f"Execution order has {len(node_order)} nodes, expected {len(self.nodes_config)}")
            return False

        if set(node_order) != set(self.nodes_config.keys()):
            missing = set(self.nodes_config.keys()) - set(node_order)
            extra = set(node_order) - set(self.nodes_config.keys())
            if missing:
                logger.error(f"Missing nodes in execution order: {missing}")
            if extra:
                logger.error(f"Extra nodes in execution order: {extra}")
            return False

        # Check that all dependencies come before dependents
        position_map = {node_id: i for i, node_id in enumerate(node_order)}

        for node_id in node_order:
            node_position = position_map[node_id]
            for dep_id in self.dependencies[node_id]:
                dep_position = position_map[dep_id]
                if dep_position >= node_position:
                    logger.error(
                        f"Dependency violation: {dep_id} (pos {dep_position}) "
                        f"must come before {node_id} (pos {node_position})"
                    )
                    return False

        return True

    def _validate_single_node(
        self, node_id: str, mock_output_manager: MockOutputManager
    ) -> NodeValidationInfo:
        """
        Validate a single node and return its validation info.

        Args:
            node_id: The ID of the node to validate
            mock_output_manager: Manager for mock dependency outputs

        Returns:
            NodeValidationInfo containing the validation results
        """
        node_type = self.nodes_config[node_id]["type"]

        try:
            # Instantiate the node
            node_instance = self._instantiate_node(node_id)
            self.node_instances[node_id] = node_instance

            # Prepare mock input data from dependencies
            input_data_for_validation = mock_output_manager.get_mock_input_for_node(
                node_id
            )

            # Validate the node
            validation_result = node_instance.validate_configuration(
                input_data=input_data_for_validation
            )

            # Convert ValidationResult to NodeValidationInfo
            errors = []
            for error in validation_result.errors:
                errors.append(
                    {
                        "type": error.error_type,
                        "message": error.message,
                        "field": error.field,
                        "value": str(error.value) if error.value is not None else None,
                    }
                )

            warnings = []
            for warning in validation_result.warnings:
                warnings.append(
                    {
                        "type": warning.error_type,
                        "message": warning.message,
                        "field": warning.field,
                        "value": str(warning.value)
                        if warning.value is not None
                        else None,
                    }
                )

            return NodeValidationInfo(
                node_id=node_id,
                node_type=node_type,
                is_valid=validation_result.is_valid(),
                has_warnings=validation_result.has_warnings(),
                summary=validation_result.get_summary(),
                detailed_report=validation_result.get_detailed_report(),
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            logger.critical(
                f"Node '{node_id}': Exception during validation: {e}", exc_info=True
            )
            return NodeValidationInfo(
                node_id=node_id,
                node_type=node_type,
                is_valid=False,
                summary=f"Validation failed with exception: {e}",
                detailed_report=f"Exception during validation: {e}",
                errors=[
                    {
                        "type": "validation_exception",
                        "message": str(e),
                        "field": None,
                        "value": None,
                    }
                ],
            )

    def _log_node_validation_result(self, node_info: NodeValidationInfo) -> None:
        """
        Log the validation outcome for a single node.

        Args:
            node_info: The validation info for the node
        """
        if node_info.is_valid:
            logger.info(f"Node '{node_info.node_id}': {node_info.summary}")
        else:
            logger.error(f"Node '{node_info.node_id}': {node_info.summary}")
            if node_info.detailed_report != node_info.summary:
                logger.error(
                    f"Node '{node_info.node_id}': Detailed validation report:\n{node_info.detailed_report}"
                )

        if node_info.has_warnings:
            logger.warning(f"Node '{node_info.node_id}': Has warnings")
            for warning in node_info.warnings:
                logger.warning(
                    f"Node '{node_info.node_id}': {warning.get('type')}: {warning.get('message')}"
                )

    def _log_final_summary(self, report: WorkflowValidationReport) -> None:
        """
        Log the final validation summary.

        Args:
            report: The complete validation report
        """
        if report.is_overall_valid:
            logger.info(f"--- Workflow Validation PASSED: '{report.workflow_name}' ---")
            logger.info(f"All {report.valid_nodes_count} nodes are valid")
            if report.total_warnings > 0:
                logger.info(f"Total warnings: {report.total_warnings}")
        else:
            logger.error(
                f"--- Workflow Validation FAILED: '{report.workflow_name}' ---"
            )
            logger.error(f"Invalid nodes: {len(report.nodes_with_errors)}")
            logger.error(f"Total errors: {report.total_errors}")
            logger.error(f"Nodes with errors: {report.nodes_with_errors}")

    def validate_workflow(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate all nodes in the workflow before execution.

        Returns:
            Tuple of (is_valid, validation_report)
            - is_valid: True if no validation errors, False otherwise
            - validation_report: Dictionary containing detailed validation results
        """
        logger.info(f"--- Starting Workflow Validation: '{self.workflow_name}' ---")

        # Initialize validation report and mock output manager
        report = WorkflowValidationReport(
            workflow_name=self.workflow_name, total_nodes=len(self.execution_order)
        )
        mock_output_manager = MockOutputManager(self)

        # Validate nodes in execution order
        for i, node_id in enumerate(self.execution_order):
            if node_id.startswith("__ENGINE_CLEANUP__"):
                logger.info(f"--- [{i + 1}/{report.total_nodes}] Identified Engine Cleanup Point: '{node_id}' ---")
                continue

            logger.info(
                f"--- [{i + 1}/{report.total_nodes}] Validating Node: '{node_id}' ---"
            )

            # Validate the individual node
            node_info = self._validate_single_node(node_id, mock_output_manager)
            report.add_node_result(node_info)

            # Log the validation result
            self._log_node_validation_result(node_info)

            # Add mock output for this node (for dependents' validation)
            mock_output_manager.add_mock_output(node_id)

        # Log final summary
        self._log_final_summary(report)

        return report.is_overall_valid, report.to_dict()

    def run(self, validate_first: bool = True):
        """
        Executes the workflow nodes in the determined topological order.

        Args:
            validate_first: If True, validates all nodes before execution
        """
        logger.info(f"--- Starting Workflow Execution: '{self.workflow_name}' ---")
        
        # Show execution tree at the start
        if self.execution_order:
            logger.info("")
            ascii_tree = generate_execution_tree_ascii(self.execution_order, self.nodes_config, self.dependencies)
            logger.info(ascii_tree)
            logger.info("")

        # Enable deferred cleanup for engine sharing if optimization is enabled
        if self.optimize_for_engines:
            try:
                from polysome.engines.engine_pool import get_engine_pool

                engine_pool = get_engine_pool()
                engine_pool.set_defer_cleanup(True)
                logger.info("Enabled deferred cleanup for engine sharing optimization")
            except Exception as e:
                logger.warning(f"Could not enable deferred cleanup: {e}")

        # Optional validation phase
        if validate_first:
            is_valid, validation_report = self.validate_workflow()

            if not is_valid:
                error_msg = (
                    f"Workflow validation failed with {validation_report['summary']['total_errors']} errors. "
                    f"Execution aborted. Check logs for detailed validation report."
                )
                logger.error(error_msg)
                raise WorkflowValidationError(error_msg)

            logger.info("Workflow validation passed. Proceeding with execution...")

        self.node_outputs = {}  # Clear previous run outputs if any

        actual_node_count = len([item for item in self.execution_order if not item.startswith("__ENGINE_CLEANUP__")])
        node_counter = 0
        
        # Track overall success
        all_nodes_successful = True

        for i, item in enumerate(self.execution_order):
            # Check if this is a cleanup marker
            if item.startswith("__ENGINE_CLEANUP__"):
                self._process_engine_cleanup_marker(item)
                continue
            
            # This is a regular node
            node_id = item
            node_counter += 1
            logger.info(f"--- [{node_counter}/{actual_node_count}] Executing Node: '{node_id}' ---")

            # Log engine pool status if applicable
            try:
                from polysome.engines.engine_pool import get_engine_pool

                engine_pool = get_engine_pool()
                stats = engine_pool.get_engine_stats()
                if stats:
                    logger.debug(f"Engine pool status: {len(stats)} engines loaded")
                    for _, engine_stat in stats.items():
                        logger.debug(
                            f"  {engine_stat['engine_name']} '{engine_stat['model_name']}': {engine_stat['reference_count']} refs"
                        )
            except Exception as e:
                logger.debug(f"Could not log engine pool status: {e}")

            # --- Prepare Input Data from Dependencies ---
            input_data_for_node: Dict[str, Any] = {}
            node_dependencies = self.dependencies[node_id]
            all_deps_met = True
            for dep_id in node_dependencies:
                if dep_id not in self.node_outputs:
                    logger.error(
                        f"Critical internal error: Dependency '{dep_id}' for node '{node_id}' was expected to run but has no output recorded. Aborting workflow."
                    )
                    all_deps_met = False
                    break
                input_data_for_node[dep_id] = self.node_outputs[dep_id]

            if not all_deps_met:
                logger.error(
                    f"Workflow execution aborted due to unmet dependency for node '{node_id}'."
                )
                all_nodes_successful = False
                break

            # --- Run Node ---
            try:
                # Use existing instance if available (from validation), otherwise create new one
                if node_id not in self.node_instances:
                    node_instance = self._instantiate_node(node_id)
                    self.node_instances[node_id] = node_instance
                else:
                    node_instance = self.node_instances[node_id]

                # Run the node
                output_info = node_instance.run(input_data=input_data_for_node)

                # Store the output information
                self.node_outputs[node_id] = output_info
                logger.info(
                    f"Node '{node_id}' finished with status: {output_info.get('status', 'unknown')}"
                )

                if output_info.get("status", "").startswith("failed"):
                    logger.warning(
                        f"Node '{node_id}' reported a failure. Subsequent nodes may be affected or fail."
                    )
                    all_nodes_successful = False

                # Clean up node resources (like models) after execution
                # JSONLProcessingNode already calls cleanup_processing in its finally block,
                # but other node types might not, so we ensure it gets called
                try:
                    # Check if this is NOT a JSONLProcessingNode (which handles its own cleanup)
                    from polysome.nodes.jsonl_processing_node import (
                        JSONLProcessingNode,
                    )

                    if not isinstance(node_instance, JSONLProcessingNode):
                        node_instance.cleanup_processing()
                        logger.debug(
                            f"Node '{node_id}': Called cleanup_processing for resource cleanup"
                        )
                except Exception as e:
                    logger.warning(
                        f"Node '{node_id}': Error during cleanup (ignored): {e}"
                    )

            except Exception as e:
                logger.critical(
                    f"Node '{node_id}' raised an unhandled exception during execution: {e}",
                    exc_info=True,
                )
                self.node_outputs[node_id] = {
                    "status": "failed_exception",
                    "error": str(e),
                }
                logger.error(
                    f"Workflow execution aborted due to critical error in node '{node_id}'."
                )
                all_nodes_successful = False
                break

        logger.info(f"--- Workflow Execution Finished: '{self.workflow_name}' ---")

        # Clean up any remaining engines in the pool
        try:
            from polysome.engines.engine_pool import get_engine_pool

            engine_pool = get_engine_pool()
            stats = engine_pool.get_engine_stats()
            if stats:
                logger.info(f"Cleaning up {len(stats)} remaining engines from pool")
                engine_pool.cleanup_all_engines()
            else:
                logger.debug("No engines remaining in pool to clean up")
        except Exception as e:
            logger.warning(f"Error during engine pool cleanup: {e}")

        return all_nodes_successful

    def print_execution_tree(self) -> str:
        """
        Generate and return the ASCII execution tree for the current workflow.
        
        Returns:
            ASCII tree representation of the execution order
        """
        if not self.execution_order:
            return "No execution order available. Run workflow setup first."
        
        return generate_execution_tree_ascii(self.execution_order, self.nodes_config, self.dependencies)

    def get_log_dir(self) -> Path:
        """
        Get the log directory for this workflow.
        
        Returns:
            Path to the log directory
        """
        return self.log_dir

    def get_workflow_name(self) -> str:
        """
        Get the workflow name for logging purposes.
        
        Returns:
            Sanitized workflow name suitable for filenames
        """
        return self.workflow_name
