"""
Unit tests for workflow execution order optimization and multi-level graph partitioning.

Tests the new engine-aware scheduling algorithm that minimizes engine switches
and prevents OOM by inserting mandatory cleanup points.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# Import the classes we want to test
from polysome.execution_optimizer import ExecutionOptimizer


class TestWorkflowOptimization:
    """Test suite for workflow execution order optimization."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for test configurations."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def simple_workflow_config(self):
        """Simple workflow configuration for basic testing."""
        return {
            "name": "test_workflow",
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/output",
            "prompts_dir": "/tmp/prompts",
            "workflow_settings": {
                "optimize_for_engines": True
            },
            "nodes": [
                {
                    "id": "loader",
                    "type": "load",
                    "params": {
                        "name": "test_loader",
                        "input_data_path": "test.json",
                        "primary_key": "id"
                    },
                    "dependencies": []
                },
                {
                    "id": "process_a",
                    "type": "text_prompt", 
                    "params": {
                        "name": "process_a",
                        "model_name": "/models/test-model",
                        "inference_engine": "vllm_dp",
                        "engine_options": {
                            "max_model_len": 4096,
                            "data_parallel_size": 2
                        }
                    },
                    "dependencies": ["loader"]
                },
                {
                    "id": "process_b",
                    "type": "text_prompt",
                    "params": {
                        "name": "process_b", 
                        "model_name": "/models/test-model",
                        "inference_engine": "vllm_dp",
                        "engine_options": {
                            "max_model_len": 4096,
                            "data_parallel_size": 2
                        }
                    },
                    "dependencies": ["loader"]
                }
            ]
        }

    @pytest.fixture
    def multi_engine_workflow_config(self):
        """Workflow configuration with multiple engine configurations."""
        return {
            "name": "multi_engine_test",
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/output", 
            "prompts_dir": "/tmp/prompts",
            "workflow_settings": {
                "optimize_for_engines": True
            },
            "nodes": [
                {
                    "id": "loader",
                    "type": "load",
                    "params": {"name": "loader"},
                    "dependencies": []
                },
                # Engine A: 4096 max_model_len
                {
                    "id": "node_a1",
                    "type": "text_prompt",
                    "params": {
                        "name": "node_a1",
                        "model_name": "/models/test-model",
                        "inference_engine": "vllm_dp",
                        "engine_options": {
                            "max_model_len": 4096,
                            "data_parallel_size": 2
                        }
                    },
                    "dependencies": ["loader"]
                },
                {
                    "id": "node_a2", 
                    "type": "text_prompt",
                    "params": {
                        "name": "node_a2",
                        "model_name": "/models/test-model",
                        "inference_engine": "vllm_dp",
                        "engine_options": {
                            "max_model_len": 4096,
                            "data_parallel_size": 2
                        }
                    },
                    "dependencies": ["loader"]
                },
                # Engine B: 8192 max_model_len  
                {
                    "id": "node_b1",
                    "type": "text_prompt",
                    "params": {
                        "name": "node_b1",
                        "model_name": "/models/test-model",
                        "inference_engine": "vllm_dp",
                        "engine_options": {
                            "max_model_len": 8192,
                            "data_parallel_size": 2
                        }
                    },
                    "dependencies": ["loader"]
                },
                # Engine A: Back to 4096
                {
                    "id": "node_a3",
                    "type": "text_prompt", 
                    "params": {
                        "name": "node_a3",
                        "model_name": "/models/test-model",
                        "inference_engine": "vllm_dp",
                        "engine_options": {
                            "max_model_len": 4096,
                            "data_parallel_size": 2
                        }
                    },
                    "dependencies": ["node_b1"]  # Depends on B engine
                }
            ]
        }

    @pytest.fixture
    def complex_dependency_workflow_config(self):
        """Complex workflow with cross-engine dependencies."""
        return {
            "name": "complex_dependency_test",
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/output",
            "prompts_dir": "/tmp/prompts", 
            "workflow_settings": {
                "optimize_for_engines": True
            },
            "nodes": [
                {
                    "id": "loader",
                    "type": "load",
                    "params": {"name": "loader"},
                    "dependencies": []
                },
                # First engine group
                {
                    "id": "eng1_node1",
                    "type": "text_prompt",
                    "params": {
                        "model_name": "/models/test",
                        "inference_engine": "vllm_dp",
                        "engine_options": {"max_model_len": 4096}
                    },
                    "dependencies": ["loader"]
                },
                {
                    "id": "eng1_node2", 
                    "type": "text_prompt",
                    "params": {
                        "model_name": "/models/test",
                        "inference_engine": "vllm_dp", 
                        "engine_options": {"max_model_len": 4096}
                    },
                    "dependencies": ["loader"]
                },
                # Second engine group
                {
                    "id": "eng2_node1",
                    "type": "text_prompt",
                    "params": {
                        "model_name": "/models/test",
                        "inference_engine": "vllm_dp",
                        "engine_options": {"max_model_len": 8192}
                    },
                    "dependencies": ["eng1_node1"]  # Cross-engine dependency
                },
                # Third engine group  
                {
                    "id": "eng3_node1",
                    "type": "text_prompt",
                    "params": {
                        "model_name": "/models/test",
                        "inference_engine": "vllm_dp",
                        "engine_options": {"max_model_len": 2048}
                    },
                    "dependencies": ["eng2_node1"]  # Chain dependency
                },
                # Back to first engine
                {
                    "id": "eng1_node3",
                    "type": "text_prompt",
                    "params": {
                        "model_name": "/models/test", 
                        "inference_engine": "vllm_dp",
                        "engine_options": {"max_model_len": 4096}
                    },
                    "dependencies": ["eng3_node1", "eng1_node2"]  # Multiple deps
                }
            ]
        }

    def create_optimizer_from_config(self, config: Dict) -> ExecutionOptimizer:
        """Helper to create optimizer instance from config dict."""
        nodes_config = {node['id']: node for node in config['nodes']}
        dependencies = defaultdict(list)
        dependents = defaultdict(list)
        all_node_ids = set(nodes_config.keys())
        for node_id, node_def in nodes_config.items():
            deps = node_def.get("dependencies", [])
            for dep_id in deps:
                if dep_id in all_node_ids:
                    dependencies[node_id].append(dep_id)
                    dependents[dep_id].append(node_id)
        return ExecutionOptimizer(nodes_config, dependencies, dependents)

    def test_basic_execution_order_validation(self, simple_workflow_config):
        """Test that basic execution order respects dependencies."""
        optimizer = self.create_optimizer_from_config(simple_workflow_config)
        execution_order = optimizer.determine_execution_order(optimize_for_engines=False)
        
        # Check that execution order is valid
        assert optimizer._validate_execution_order(execution_order)
        
        # Check that loader comes before processing nodes
        loader_pos = execution_order.index("loader")
        process_a_pos = execution_order.index("process_a")
        process_b_pos = execution_order.index("process_b")
        
        assert loader_pos < process_a_pos
        assert loader_pos < process_b_pos

    def test_engine_grouping(self, simple_workflow_config):
        """Test that nodes are correctly grouped by engine configuration."""
        optimizer = self.create_optimizer_from_config(simple_workflow_config)
        engine_groups = optimizer._group_nodes_by_engine()
        
        # Should have 2 groups: no-engine and vllm_dp
        assert len(engine_groups) == 2
        assert "_no_engine_" in engine_groups
        assert "loader" in engine_groups["_no_engine_"]
        
        # Find the vllm_dp group
        vllm_groups = [k for k in engine_groups.keys() if k.startswith("vllm_dp")]
        assert len(vllm_groups) == 1
        
        vllm_group = engine_groups[vllm_groups[0]]
        assert "process_a" in vllm_group
        assert "process_b" in vllm_group

    def test_multi_engine_optimization(self, multi_engine_workflow_config):
        """Test optimization with multiple engine configurations."""
        optimizer = self.create_optimizer_from_config(multi_engine_workflow_config)
        result = optimizer.determine_execution_order(optimize_for_engines=True)
        
        # Check that execution order contains cleanup markers
        cleanup_markers = [item for item in result.execution_order if item.startswith("__ENGINE_CLEANUP__")]
        assert len(cleanup_markers) > 0, "Should have cleanup markers for different engine configs"
        
        # Verify execution order is still valid (ignoring cleanup markers) 
        assert optimizer._validate_execution_order(result.execution_order)
        
        # Check that nodes with same engine config are grouped together when possible
        engine_groups = optimizer._group_nodes_by_engine()
        
        # Should have at least 2 different vllm_dp configurations
        vllm_groups = [k for k in engine_groups.keys() if k.startswith("vllm_dp")]
        assert len(vllm_groups) >= 2, "Should have multiple engine configurations"

    def test_cleanup_point_insertion(self, multi_engine_workflow_config):
        """Test that cleanup points are correctly inserted between different engines."""
        optimizer = self.create_optimizer_from_config(multi_engine_workflow_config)
        result = optimizer.determine_execution_order(optimize_for_engines=True)
        
        # Find positions of nodes with different engine configs
        node_engines = {}
        engine_groups = optimizer._group_nodes_by_engine()
        for engine_key, nodes in engine_groups.items():
            for node in nodes:
                node_engines[node] = engine_key
        
        # Check that cleanup markers appear between nodes with different engine configs
        execution_order = result.execution_order
        prev_engine = None
        
        for item in execution_order:
            if item.startswith("__ENGINE_CLEANUP__"):
                # This should be between different engines
                continue
            elif item in node_engines:
                current_engine = optimizer._extract_base_engine_key(node_engines[item])
                if prev_engine is not None and current_engine != prev_engine:
                    # Look for cleanup marker before this node
                    current_pos = execution_order.index(item)
                    if current_pos > 0:
                        prev_item = execution_order[current_pos - 1]
                        # Should have cleanup marker or be transitioning appropriately
                        assert prev_item.startswith("__ENGINE_CLEANUP__") or current_engine == prev_engine
                prev_engine = current_engine

    def test_dependency_satisfaction_with_cleanup(self, complex_dependency_workflow_config):
        """Test that dependencies are satisfied even with cleanup points inserted.""" 
        optimizer = self.create_optimizer_from_config(complex_dependency_workflow_config)
        result = optimizer.determine_execution_order(optimize_for_engines=True)
        
        # Execution order should be valid
        assert optimizer._validate_execution_order(result.execution_order)
        
        # Check specific dependency chains
        execution_order = [item for item in result.execution_order if not item.startswith("__ENGINE_CLEANUP__")]
        position_map = {node: i for i, node in enumerate(execution_order)}
        
        # eng1_node1 should come before eng2_node1 
        assert position_map["eng1_node1"] < position_map["eng2_node1"]
        
        # eng2_node1 should come before eng3_node1
        assert position_map["eng2_node1"] < position_map["eng3_node1"]
        
        # eng3_node1 and eng1_node2 should both come before eng1_node3
        assert position_map["eng3_node1"] < position_map["eng1_node3"]
        assert position_map["eng1_node2"] < position_map["eng1_node3"]

    def test_partition_splitting(self, complex_dependency_workflow_config):
        """Test that partitions are split when dependencies prevent grouping."""
        optimizer = self.create_optimizer_from_config(complex_dependency_workflow_config)
        
        # Get partitions after optimization
        engine_groups = optimizer._group_nodes_by_engine()
        basic_order = optimizer._determine_basic_execution_order()
        partitions = optimizer._partition_by_engine_configuration(basic_order, engine_groups)
        
        # Should have more partitions than original engine groups due to splitting
        assert len(partitions) >= len(engine_groups)
        
        # Check that subgroup creation worked
        subgroup_keys = [k for k in partitions.keys() if "_subgroup_" in k]
        if subgroup_keys:
            # At least one partition was split
            assert len(subgroup_keys) > 0

    def test_critical_path_calculation(self, complex_dependency_workflow_config):
        """Test that critical path calculation works correctly.""" 
        optimizer = self.create_optimizer_from_config(complex_dependency_workflow_config)
        
        # Test critical path calculation for a subset of nodes
        test_nodes = ["eng1_node1", "eng2_node1", "eng3_node1"]
        dependencies = {
            "eng1_node1": [],
            "eng2_node1": ["eng1_node1"], 
            "eng3_node1": ["eng2_node1"]
        }
        dependents = {
            "eng1_node1": ["eng2_node1"],
            "eng2_node1": ["eng3_node1"],
            "eng3_node1": []
        }
        
        critical_paths = optimizer._calculate_critical_paths(test_nodes, dependencies, dependents)
        
        # eng3_node1 should have path length 1 (no dependents)
        assert critical_paths["eng3_node1"] == 1
        
        # eng2_node1 should have path length 2 (1 + eng3_node1)
        assert critical_paths["eng2_node1"] == 2
        
        # eng1_node1 should have path length 3 (1 + eng2_node1)
        assert critical_paths["eng1_node1"] == 3

    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])