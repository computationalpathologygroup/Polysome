import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


class ExecutionOptimizer:
    """
    Analyzes a workflow's Directed Acyclic Graph (DAG) to determine the
    optimal execution order for nodes, with a focus on minimizing engine
    loading and unloading to improve performance and reduce memory usage.
    """

    def __init__(self, nodes_config: Dict[str, Dict], dependencies: Dict[str, List[str]], dependents: Dict[str, List[str]]):
        """
        Initializes the ExecutionOptimizer.

        Args:
            nodes_config: Configuration for all nodes in the workflow.
            dependencies: A dictionary mapping each node_id to a list of its dependency_ids.
            dependents: A dictionary mapping each node_id to a list of its dependent_ids.
        """
        self.nodes_config = nodes_config
        self.dependencies = dependencies
        self.dependents = dependents
        self.execution_order: List[str] = []

    def determine_execution_order(self, optimize_for_engines: bool = True) -> List[str]:
        """
        Determines a valid execution order using topological sort.

        Args:
            optimize_for_engines: If True, optimize order to minimize model loading/unloading.

        Returns:
            A list of node IDs in the determined execution order.
        """
        if optimize_for_engines:
            self.execution_order, self.cleanup_points, self.engine_groups, self.basic_order = self._determine_optimized_execution_order()
        else:
            self.execution_order = self._determine_basic_execution_order()
            self.cleanup_points = []
            self.engine_groups = {}
            self.basic_order = self.execution_order

        logger.info(f"Determined execution order: {self.execution_order}")
        return self

    def _determine_basic_execution_order(self) -> List[str]:
        """Basic topological sort using Kahn's algorithm."""
        in_degree = {
            node_id: len(self.dependencies[node_id]) for node_id in self.nodes_config
        }
        queue = deque(
            sorted(
                [node_id for node_id in self.nodes_config if in_degree[node_id] == 0]
            )
        )  # Sort initial nodes for determinism
        execution_order = []

        while queue:
            u = queue.popleft()
            execution_order.append(u)

            # Process dependents in sorted order for deterministic runs if structure allows
            for v in sorted(self.dependents[u]):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        return execution_order

    def _determine_optimized_execution_order(self) -> Tuple[List[str], List[Dict], Dict, List[str]]:
        """
        Engine-optimized topological sort that groups nodes by engine usage
        to minimize model loading/unloading operations.
        """
        # First, get a basic valid topological order
        basic_order = self._determine_basic_execution_order()

        # Group nodes by their engine configuration
        engine_groups = self._group_nodes_by_engine()

        if not engine_groups or len(engine_groups) <= 1:
            # No optimization possible or needed
            logger.info("No engine optimization needed - using basic execution order")
            return basic_order, [], engine_groups, basic_order

        # Optimize the order while maintaining topological constraints
        optimized_order_with_cleanup = self._optimize_order_for_engines(basic_order, engine_groups)

        # Validate that optimized order still respects dependencies
        if self._validate_execution_order(optimized_order_with_cleanup):
            logger.info(
                f"Applied engine optimization - grouped {len(engine_groups)} engine types"
            )
            cleanup_points = self._get_cleanup_points(optimized_order_with_cleanup)
            return optimized_order_with_cleanup, cleanup_points, engine_groups, basic_order
        else:
            logger.warning("Engine optimization failed validation - using basic order")
            return basic_order, [], engine_groups, basic_order

    def _group_nodes_by_engine(self) -> Dict[str, List[str]]:
        """
        Group nodes by their engine configuration, separating source and sink no-engine nodes.
        """
        engine_groups = defaultdict(list)
        
        # First pass: identify engine-specific nodes and initial no-engine candidates
        no_engine_candidates = []
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
                no_engine_candidates.append(node_id)

        # Second pass: classify no-engine nodes as source or sink
        engine_node_ids = set(node_id for key, nodes in engine_groups.items() if key != "_no_engine_" for node_id in nodes)

        for node_id in no_engine_candidates:
            is_sink = False

            # Check if it's a sink node (depends on any engine node)
            for dep_id in self.dependencies[node_id]:
                if dep_id in engine_node_ids:
                    is_sink = True
                    break
            
            if is_sink:
                engine_groups["_no_engine_sink_"].append(node_id)
            else:
                engine_groups["_no_engine_source_"].append(node_id)

        return engine_groups

    def _optimize_order_for_engines(
        self, basic_order: List[str], engine_groups: Dict[str, List[str]]
    ) -> List[str]:
        """
        Optimize execution order using multi-level graph partitioning to minimize
        engine switches while ensuring proper cleanup between different configurations.
        """
        logger.info("Starting multi-level graph partitioning for engine optimization")
        
        optimized_partitions = self._partition_by_engine_configuration(basic_order, engine_groups)
        
        for partition_key, partition_nodes in optimized_partitions.items():
            optimized_partitions[partition_key] = self._optimize_within_partition(partition_nodes)
        
        final_order = self._schedule_partitions_with_cleanup(optimized_partitions)
        
        logger.info(f"Multi-level optimization complete. Final order: {len(final_order)} nodes across {len(optimized_partitions)} engine partitions")
        return final_order

    def _partition_by_engine_configuration(
        self, basic_order: List[str], engine_groups: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Partition DAG by engine configuration to minimize cross-partition dependencies.
        """
        logger.debug("Phase 1: Starting multi-level graph partitioning by engine configuration")
        
        optimized_partitions = {}
        
        for engine_key, nodes in engine_groups.items():
            if not nodes:
                continue
            
            if self._can_execute_partition_together(nodes):
                optimized_partitions[engine_key] = nodes
                logger.debug(f"Partition '{engine_key}' can be executed as single group")
            else:
                subpartitions = self._split_partition_by_dependencies(engine_key, nodes)
                for i, subpartition in enumerate(subpartitions):
                    subkey = f"{engine_key}_subgroup_{i}"
                    optimized_partitions[subkey] = subpartition
                    logger.debug(f"Created subpartition '{subkey}' with {len(subpartition)} nodes")
        
        return optimized_partitions

    def _get_node_engine_key(self, node_id: str, engine_groups: Dict[str, List[str]]) -> str:
        """Helper to find which engine group a node belongs to."""
        for engine_key, nodes in engine_groups.items():
            if node_id in nodes:
                return engine_key
        return "_no_engine_"

    def _can_execute_partition_together(self, nodes: List[str]) -> bool:
        """
        Check if all nodes in a partition can be executed together.
        """
        node_set = set(nodes)
        
        for node_id in nodes:
            for dep_id in self.dependencies[node_id]:
                if dep_id not in node_set:
                    for other_node in nodes:
                        if dep_id in self.dependencies[other_node]:
                            continue
                        other_deps = set(self.dependencies[other_node])
                        if other_deps.intersection(node_set) and dep_id not in other_deps:
                            return False
        
        return True

    def _split_partition_by_dependencies(self, engine_key: str, nodes: List[str]) -> List[List[str]]:
        """
        Split an engine partition into subgroups based on dependency conflicts.
        """
        logger.debug(f"Splitting partition '{engine_key}' with {len(nodes)} nodes")
        
        subpartitions = []
        remaining_nodes = nodes.copy()
        
        while remaining_nodes:
            current_subpartition = []
            nodes_to_remove = []
            
            for node_id in remaining_nodes:
                can_add = True
                for dep_id in self.dependencies[node_id]:
                    if dep_id in remaining_nodes and dep_id not in current_subpartition:
                        can_add = False
                        break
                
                if can_add:
                    current_subpartition.append(node_id)
                    nodes_to_remove.append(node_id)
            
            for node_id in nodes_to_remove:
                remaining_nodes.remove(node_id)
            
            if current_subpartition:
                subpartitions.append(current_subpartition)
            else:
                if remaining_nodes:
                    subpartitions.append([remaining_nodes.pop(0)])
        
        logger.debug(f"Split into {len(subpartitions)} subpartitions")
        return subpartitions

    def _optimize_within_partition(self, partition_nodes: List[str]) -> List[str]:
        """
        Intra-partition optimization using topological sort + critical path analysis.
        """
        if not partition_nodes or len(partition_nodes) == 1:
            return partition_nodes
        
        logger.debug(f"Phase 2: Optimizing partition with {len(partition_nodes)} nodes")
        
        partition_set = set(partition_nodes)
        partition_dependencies = {n: [d for d in self.dependencies[n] if d in partition_set] for n in partition_nodes}
        partition_dependents = {n: [d for d in self.dependents[n] if d in partition_set] for n in partition_nodes}
        
        critical_paths = self._calculate_critical_paths(partition_nodes, partition_dependencies, partition_dependents)
        
        optimized_order = self._topological_sort_with_priority(
            partition_nodes, partition_dependencies, critical_paths
        )
        
        logger.debug(f"Intra-partition optimization complete: {len(optimized_order)} nodes ordered")
        return optimized_order

    def _calculate_critical_paths(
        self, nodes: List[str], dependencies: Dict[str, List[str]], dependents: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """
        Calculate critical path lengths for nodes using dynamic programming.
        """
        critical_paths = {}
        visited = set()
        
        def calculate_path_length(node_id: str) -> int:
            if node_id in visited:
                return critical_paths.get(node_id, 0)
            
            visited.add(node_id)
            
            if not dependents.get(node_id, []):
                critical_paths[node_id] = 1
                return 1
            
            max_dependent_path = 0
            for dependent in dependents[node_id]:
                if dependent in nodes:
                    max_dependent_path = max(max_dependent_path, calculate_path_length(dependent))
            
            critical_paths[node_id] = 1 + max_dependent_path
            return critical_paths[node_id]
        
        for node_id in nodes:
            calculate_path_length(node_id)
        
        return critical_paths

    def _topological_sort_with_priority(
        self, nodes: List[str], dependencies: Dict[str, List[str]], critical_paths: Dict[str, int]
    ) -> List[str]:
        """
        Perform topological sort with critical path priority.
        """
        in_degree = {node_id: len(dependencies[node_id]) for node_id in nodes}
        
        import heapq
        ready_heap = [(-critical_paths[n], n) for n in nodes if in_degree[n] == 0]
        heapq.heapify(ready_heap)
        
        result = []
        
        while ready_heap:
            _, current_node = heapq.heappop(ready_heap)
            result.append(current_node)
            
            for dependent in self.dependents.get(current_node, []):
                if dependent in nodes:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        heapq.heappush(ready_heap, (-critical_paths[dependent], dependent))
        
        if len(result) != len(nodes):
            logger.warning(f"Topological sort incomplete: got {len(result)} nodes, expected {len(nodes)}")
            result.extend(sorted(set(nodes) - set(result)))
        
        return result

    def _schedule_partitions_with_cleanup(self, partitions: Dict[str, List[str]]) -> List[str]:
        """
        Inter-partition scheduling with mandatory cleanup points.
        """
        logger.debug(f"Phase 3: Scheduling {len(partitions)} partitions with mandatory cleanup")
        
        if not partitions:
            return []
        
        if len(partitions) == 1:
            return list(partitions.values())[0]
        
        partition_dependencies = self._build_partition_dependency_graph(partitions)
        partition_order = self._topological_sort_partitions(partitions, partition_dependencies)
        
        final_order = []
        previous_engine_key = None
        
        for partition_key in partition_order:
            current_engine_key = self._extract_base_engine_key(partition_key)
            partition_nodes = partitions[partition_key]
            
            if previous_engine_key is not None and previous_engine_key != current_engine_key:
                cleanup_marker = f"__ENGINE_CLEANUP__{previous_engine_key}_TO_{current_engine_key}__"
                final_order.append(cleanup_marker)
                logger.debug(f"Inserted cleanup point: {previous_engine_key} → {current_engine_key}")
            
            final_order.extend(partition_nodes)
            previous_engine_key = current_engine_key
        
        return final_order

    def _build_partition_dependency_graph(self, partitions: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Build dependency graph between partitions.
        """
        partition_deps = {key: [] for key in partitions.keys()}
        node_to_partition = {node_id: key for key, nodes in partitions.items() for node_id in nodes}
        
        logger.debug("Building partition dependency graph:")
        for partition_key, nodes in partitions.items():
            dependent_partitions = set()
            for node_id in nodes:
                for dep_id in self.dependencies[node_id]:
                    dep_partition = node_to_partition.get(dep_id)
                    if dep_partition and dep_partition != partition_key:
                        dependent_partitions.add(dep_partition)
            partition_deps[partition_key] = list(dependent_partitions)
            logger.debug(f"  Partition '{partition_key}' depends on: {partition_deps[partition_key]}")
        
        return partition_deps

    def _topological_sort_partitions(
        self, partitions: Dict[str, List[str]], partition_dependencies: Dict[str, List[str]]
    ) -> List[str]:
        """
        Topologically sort partitions.
        """
        in_degree = {key: len(deps) for key, deps in partition_dependencies.items()}
        logger.debug(f"Initial partition in-degrees: {in_degree}")
        
        from collections import deque
        queue = deque([key for key, degree in in_degree.items() if degree == 0])
        logger.debug(f"Initial partition queue: {queue}")
        
        result = []
        
        while queue:
            current_partition = queue.popleft()
            result.append(current_partition)
            logger.debug(f"Processing partition: {current_partition}")
            
            for partition_key, deps in partition_dependencies.items():
                if current_partition in deps:
                    in_degree[partition_key] -= 1
                    logger.debug(f"  Decremented in-degree for {partition_key} to {in_degree[partition_key]}")
                    if in_degree[partition_key] == 0:
                        queue.append(partition_key)
                        logger.debug(f"  Added {partition_key} to queue.")
        
        if len(result) != len(partitions):
            logger.warning("Cycle detected in partition dependencies - using fallback ordering")
            missing = set(partitions.keys()) - set(result)
            result.extend(sorted(missing))
            logger.debug(f"Fallback ordering: {result}")
        
        return result

    def _extract_base_engine_key(self, partition_key: str) -> str:
        """
        Extract the base engine configuration key from partition key.
        """
        if "_subgroup_" in partition_key:
            return partition_key.split("_subgroup_")[0]
        return partition_key

    def _get_cleanup_points(self, execution_order: List[str]) -> List[Dict]:
        """Extracts cleanup points from the execution order."""
        cleanup_points = []
        for i, item in enumerate(execution_order):
            if item.startswith("__ENGINE_CLEANUP__"):
                parts = item.replace("__ENGINE_CLEANUP__", "").replace("__", "").split("_TO_")
                if len(parts) == 2:
                    from_engine, to_engine = parts
                    cleanup_points.append({
                        'position': i,
                        'from_engine': from_engine,
                        'to_engine': to_engine,
                        'marker': item
                    })
        return cleanup_points

    def _validate_execution_order(self, order: List[str]) -> bool:
        """
        Validate that an execution order respects all dependencies.
        """
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
