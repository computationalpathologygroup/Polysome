"""
Tree utilities for generating ASCII representations of workflow execution orders.
"""
import json
import logging
from typing import Dict, List
from collections import defaultdict

logger = logging.getLogger(__name__)


def generate_execution_tree_ascii(
    execution_order: List[str],
    nodes_config: Dict[str, Dict],
    dependencies: Dict[str, List[str]]
) -> str:
    """
    Generate a beautiful ASCII tree representation of the execution order.
    Shows nodes, engine groups, and cleanup points in a visual tree format.

    Args:
        execution_order: The final execution order with cleanup markers
        nodes_config: Configuration for all nodes in the workflow
        dependencies: Dictionary mapping node IDs to their dependencies

    Returns:
        ASCII tree string representation
    """
    if not execution_order:
        return "Empty execution order"

    tree_lines = []
    tree_lines.append("📋 Workflow Execution Tree:")
    tree_lines.append("│")
    
    # Track engine groups for visual organization
    engine_groups = _group_nodes_by_engine(nodes_config)
    node_to_engine = {}
    for engine_key, nodes in engine_groups.items():
        for node in nodes:
            node_to_engine[node] = engine_key
    
    current_engine = None
    node_count = 0
    cleanup_count = 0
    
    for i, item in enumerate(execution_order):
        is_last = (i == len(execution_order) - 1)
        
        if item.startswith("__ENGINE_CLEANUP__"):
            # This is a cleanup marker
            cleanup_count += 1
            parts = item.replace("__ENGINE_CLEANUP__", "").replace("__", "").split("_TO_")
            if len(parts) == 2:
                from_engine, to_engine = parts
                
                # Add cleanup visualization
                prefix = "└── " if is_last else "├── "
                tree_lines.append(f"│{prefix}🧹 CLEANUP: {format_engine_name(from_engine)} → {format_engine_name(to_engine)}")
                
                if not is_last:
                    tree_lines.append("│   │")
                
                current_engine = to_engine
            else:
                prefix = "└── " if is_last else "├── "
                tree_lines.append(f"│{prefix}🧹 CLEANUP: {item}")
                if not is_last:
                    tree_lines.append("│   │")
        else:
            # This is a regular node
            node_count += 1
            node_engine = node_to_engine.get(item, "_no_engine_")
            
            # Check if we're starting a new engine group
            if current_engine != node_engine:
                if current_engine is not None and i > 0:
                    # Add separator between engine groups
                    tree_lines.append("│   ┊")
                current_engine = node_engine
            
            # Format node with engine info
            prefix = "└── " if is_last else "├── "
            engine_display = format_engine_name(node_engine)
            
            # Add dependency indicators
            deps = dependencies.get(item, [])
            dep_indicator = f" (deps: {len(deps)})" if deps else ""
            
            tree_lines.append(f"│{prefix}📦 {item}{dep_indicator} [{engine_display}]")
            
            if not is_last:
                # Check if next item is a cleanup - if so, use different continuation
                next_item = execution_order[i + 1] if i + 1 < len(execution_order) else ""
                if next_item.startswith("__ENGINE_CLEANUP__"):
                    tree_lines.append("│   ↓")
                else:
                    tree_lines.append("│   │")
    
    # Add summary
    tree_lines.append("│")
    tree_lines.append(f"└── 📊 Summary: {node_count} nodes, {cleanup_count} cleanup points")
    
    return "\n".join(tree_lines)


def format_engine_name(engine_key: str) -> str:
    """
    Format engine key for display in tree.
    
    Args:
        engine_key: Full engine configuration key
        
    Returns:
        Human-readable engine name
    """
    if engine_key == "_no_engine_":
        return "no-engine"
    
    # Parse engine key to extract meaningful parts
    parts = engine_key.split("::")
    if len(parts) >= 2:
        engine_name = parts[0]
        model_name = parts[1].split("/")[-1]  # Get just the model name, not full path
        
        # Extract key configuration differences
        if len(parts) >= 3:
            try:
                options = json.loads(parts[2])
                
                # Show distinguishing characteristics
                key_params = []
                if 'max_model_len' in options:
                    key_params.append(f"len:{options['max_model_len']}")
                if 'data_parallel_size' in options and options['data_parallel_size'] > 1:
                    key_params.append(f"dp:{options['data_parallel_size']}")
                
                if key_params:
                    return f"{engine_name}::{model_name}({','.join(key_params)})"
                else:
                    return f"{engine_name}::{model_name}"
            except Exception as e:
                return f"{engine_name}::{model_name} (Error: {e})"
        else:
            return f"{engine_name}::{model_name}"
    else:
        return engine_key[:50] + "..." if len(engine_key) > 50 else engine_key


def _group_nodes_by_engine(nodes_config: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Group nodes by their engine configuration.

    Args:
        nodes_config: Configuration for all nodes in the workflow

    Returns:
        Dictionary mapping engine keys to lists of node IDs
    """
    engine_groups = defaultdict(list)

    for node_id, node_config in nodes_config.items():
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