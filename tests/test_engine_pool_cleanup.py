"""
Unit tests for engine pool forced cleanup functionality.

Tests the forced cleanup logic that prevents OOM by ensuring engines
never coexist when they have incompatible configurations.
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock

from polysome.engines.engine_pool import EnginePool, EngineInfo


class TestEnginePoolForcedCleanup:
    """Test suite for engine pool forced cleanup functionality."""

    @pytest.fixture(autouse=True)
    def reset_engine_pool(self):
        """Reset engine pool singleton before each test."""
        # Store original instance
        original_instance = EnginePool._instance
        
        # Reset singleton
        EnginePool._instance = None
        
        yield
        
        # Restore original instance
        EnginePool._instance = original_instance

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine for testing."""
        engine = Mock()
        engine.unload_model = Mock()
        return engine

    @pytest.fixture
    def mock_vllm_dp_engine(self):
        """Create a mock vLLM data parallel engine."""
        engine = Mock()
        engine.unload_model = Mock()
        
        # Mock coordinator with processes
        coordinator = Mock()
        coordinator.shutdown = Mock()
        coordinator.processes = [Mock(), Mock()]
        for process in coordinator.processes:
            process.is_alive.return_value = True
            process.terminate = Mock()
        
        engine.coordinator = coordinator
        return engine

    @pytest.fixture
    def populated_engine_pool(self, mock_engine, mock_vllm_dp_engine):
        """Engine pool with multiple engines loaded."""
        pool = EnginePool()
        
        # Add regular engine
        engine_info_1 = EngineInfo(
            engine=mock_engine,
            engine_name="huggingface",
            model_name="/models/test-model-1",
            engine_options={"max_tokens": 1024}
        )
        
        # Add vLLM DP engine with 4096 config
        engine_info_2 = EngineInfo(
            engine=mock_vllm_dp_engine,
            engine_name="vllm_dp", 
            model_name="/models/test-model-2",
            engine_options={"max_model_len": 4096, "data_parallel_size": 2}
        )
        
        # Add another vLLM DP engine with 8192 config
        mock_vllm_dp_engine_2 = Mock()
        mock_vllm_dp_engine_2.unload_model = Mock()
        coordinator_2 = Mock()
        coordinator_2.shutdown = Mock()
        mock_vllm_dp_engine_2.coordinator = coordinator_2
        
        engine_info_3 = EngineInfo(
            engine=mock_vllm_dp_engine_2,
            engine_name="vllm_dp",
            model_name="/models/test-model-2", 
            engine_options={"max_model_len": 8192, "data_parallel_size": 2}
        )
        
        # Generate keys and store engines
        key_1 = pool._generate_engine_key("huggingface", "/models/test-model-1", {"max_tokens": 1024})
        key_2 = pool._generate_engine_key("vllm_dp", "/models/test-model-2", {"max_model_len": 4096, "data_parallel_size": 2})
        key_3 = pool._generate_engine_key("vllm_dp", "/models/test-model-2", {"max_model_len": 8192, "data_parallel_size": 2})
        
        pool._engines[key_1] = engine_info_1
        pool._engines[key_2] = engine_info_2  
        pool._engines[key_3] = engine_info_3
        
        # Set reference counts
        pool._ref_counter.set_count(key_1, 1)
        pool._ref_counter.set_count(key_2, 2)
        pool._ref_counter.set_count(key_3, 1)
        
        return pool, {
            "key_1": key_1, "engine_1": mock_engine,
            "key_2": key_2, "engine_2": mock_vllm_dp_engine,
            "key_3": key_3, "engine_3": mock_vllm_dp_engine_2
        }

    def test_force_cleanup_between_engines_basic(self, populated_engine_pool):
        """Test basic forced cleanup between different engine configurations."""
        pool, engines = populated_engine_pool
        
        # Get initial stats
        initial_stats = pool.get_engine_stats()
        assert len(initial_stats) == 3
        
        # Force cleanup from one vLLM config to another
        from_key = engines["key_2"]  # 4096 config
        to_key = engines["key_3"]    # 8192 config
        
        pool.force_cleanup_between_engines(from_key, to_key)
        
        # Should have cleaned up engines that don't match target
        final_stats = pool.get_engine_stats()
        
        # Only the 8192 config engine should remain
        assert len(final_stats) == 1
        remaining_key = list(final_stats.keys())[0]
        assert remaining_key == engines["key_3"]
        
        # Verify unload_model was called on cleaned up engines
        engines["engine_1"].unload_model.assert_called_once()
        engines["engine_2"].coordinator.shutdown.assert_called_once()

    def test_force_cleanup_with_vllm_dp_special_handling(self, populated_engine_pool):
        """Test that vLLM DP engines get special cleanup handling."""
        pool, engines = populated_engine_pool
        
        # Force cleanup targeting huggingface engine (should cleanup vLLM DP engines)
        pool.force_cleanup_between_engines(
            engines["key_2"], 
            engines["key_1"]
        )
        
        # Verify vLLM DP coordinator shutdown was called
        engines["engine_2"].coordinator.shutdown.assert_called_once()
        engines["engine_3"].coordinator.shutdown.assert_called_once()
        
        # Regular engine should remain
        final_stats = pool.get_engine_stats()
        assert len(final_stats) == 1
        assert engines["key_1"] in final_stats

    def test_force_cleanup_with_exclude_keys(self, populated_engine_pool):
        """Test forced cleanup with excluded engine keys."""
        pool, engines = populated_engine_pool
        
        # Force cleanup but exclude one engine
        pool.force_cleanup_between_engines(
            engines["key_2"],
            engines["key_3"], 
            exclude_keys=[engines["key_1"]]
        )
        
        # Should have 2 engines remaining (target + excluded)
        final_stats = pool.get_engine_stats()
        assert len(final_stats) == 2
        assert engines["key_1"] in final_stats  # Excluded
        assert engines["key_3"] in final_stats  # Target
        
        # Only the non-excluded, non-target engine should be cleaned up
        engines["engine_2"].coordinator.shutdown.assert_called_once()
        engines["engine_1"].unload_model.assert_not_called()
        engines["engine_3"].unload_model.assert_not_called()

    def test_force_cleanup_empty_pool(self):
        """Test forced cleanup on empty engine pool."""
        pool = EnginePool()
        
        # Should not raise any errors
        pool.force_cleanup_between_engines("from_key", "to_key")
        
        # Pool should remain empty
        assert len(pool.get_engine_stats()) == 0

    def test_force_cleanup_same_engine_config(self, populated_engine_pool): 
        """Test forced cleanup when from and to engines are the same."""
        pool, engines = populated_engine_pool
        
        # Force cleanup with same from/to engine config
        pool.force_cleanup_between_engines(
            engines["key_2"],
            engines["key_2"]  # Same key
        )
        
        # Only the target engine should remain 
        final_stats = pool.get_engine_stats()
        assert len(final_stats) == 1
        assert engines["key_2"] in final_stats
        
        # Other engines should be unloaded, but target should not
        engines["engine_1"].unload_model.assert_called_once()
        engines["engine_2"].coordinator.shutdown.assert_not_called()  # Target engine
        engines["engine_3"].coordinator.shutdown.assert_called_once()

    def test_base_engine_key_extraction(self):
        """Test extraction of base engine keys for comparison."""
        pool = EnginePool()
        
        # Test various engine key formats
        test_cases = [
            ("vllm_dp::/models/test::{'max_model_len': 4096}", "vllm_dp::/models/test"),
            ("huggingface::/models/bert::{'max_tokens': 512}", "huggingface::/models/bert"),
            ("simple_key", "simple_key"),
            ("just::two::parts", "just::two")
        ]
        
        for full_key, expected_base in test_cases:
            result = pool._extract_base_engine_key_from_full_key(full_key)
            assert result == expected_base

    def test_force_cleanup_error_handling(self, populated_engine_pool):
        """Test error handling during forced cleanup.""" 
        pool, engines = populated_engine_pool
        
        # Make one engine throw an error during cleanup
        engines["engine_1"].unload_model.side_effect = Exception("Cleanup failed")
        
        # Should not raise exception, just log error
        pool.force_cleanup_between_engines(
            engines["key_1"],
            engines["key_2"]
        )
        
        # Other engines should still be cleaned up
        engines["engine_3"].coordinator.shutdown.assert_called_once()
        
        # Target engine should remain
        final_stats = pool.get_engine_stats()
        assert engines["key_2"] in final_stats

    def test_vllm_dp_process_termination_fallback(self, populated_engine_pool):
        """Test fallback process termination for vLLM DP engines."""
        pool, engines = populated_engine_pool
        
        # Make coordinator shutdown fail to trigger process termination fallback
        engines["engine_2"].coordinator.shutdown.side_effect = Exception("Shutdown failed")
        
        pool.force_cleanup_between_engines(engines["key_2"], engines["key_1"])
        
        # Should have attempted coordinator shutdown
        engines["engine_2"].coordinator.shutdown.assert_called_once()
        
        # Should have attempted process termination as fallback
        for process in engines["engine_2"].coordinator.processes:
            process.terminate.assert_called_once()

    def test_reference_count_override(self, populated_engine_pool):
        """Test that forced cleanup overrides reference counting."""
        pool, engines = populated_engine_pool
        
        # Verify engines have reference counts > 0
        initial_stats = pool.get_engine_stats()
        for stats in initial_stats.values():
            assert stats["reference_count"] > 0
        
        # Force cleanup should ignore reference counts
        pool.force_cleanup_between_engines(engines["key_1"], engines["key_2"])
        
        # Engines should be cleaned up despite having references
        final_stats = pool.get_engine_stats()
        assert len(final_stats) == 1
        assert engines["key_2"] in final_stats
        
        # Reference counts should be reset to 0 for cleaned up engines
        assert pool._ref_counter.get_count(engines["key_1"]) == 0
        assert pool._ref_counter.get_count(engines["key_3"]) == 0

    def test_concurrent_force_cleanup(self, populated_engine_pool):
        """Test thread safety of forced cleanup operations."""
        pool, engines = populated_engine_pool
        
        results = []
        errors = []
        
        def cleanup_thread(from_key, to_key):
            try:
                pool.force_cleanup_between_engines(from_key, to_key)
                results.append("success")
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple cleanup threads
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=cleanup_thread,
                args=(engines["key_1"], engines["key_2"])
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Should not have any errors
        assert len(errors) == 0, f"Concurrent cleanup errors: {errors}"
        assert len(results) == 3
        
        # Final state should be consistent
        final_stats = pool.get_engine_stats()
        assert len(final_stats) <= 1  # At most the target engine

    def test_force_cleanup_logging(self, populated_engine_pool):
        """Test that forced cleanup produces appropriate log messages."""
        pool, engines = populated_engine_pool
        
        with patch('polysome.engines.engine_pool.logger') as mock_logger:
            pool.force_cleanup_between_engines(engines["key_1"], engines["key_2"])
            
            # Should have logged the cleanup operation
            mock_logger.info.assert_any_call(f"Forcing engine cleanup: {engines['key_1']} → {engines['key_2']}")
            
            # Should have logged individual engine cleanups
            cleanup_calls = [call for call in mock_logger.info.call_args_list 
                           if "Force unloading engine:" in str(call)]
            assert len(cleanup_calls) > 0

    @pytest.mark.parametrize("engine_count", [1, 5, 10])
    def test_force_cleanup_performance(self, engine_count):
        """Test performance of forced cleanup with varying engine counts."""
        pool = EnginePool()
        
        # Add multiple engines
        engines = []
        for i in range(engine_count):
            mock_engine = Mock()
            mock_engine.unload_model = Mock()
            
            engine_info = EngineInfo(
                engine=mock_engine,
                engine_name="test_engine",
                model_name=f"/models/test-{i}",
                engine_options={"id": i}
            )
            
            key = pool._generate_engine_key("test_engine", f"/models/test-{i}", {"id": i})
            pool._engines[key] = engine_info
            pool._ref_counter.set_count(key, 1)
            engines.append((key, mock_engine))
        
        # Measure cleanup time
        start_time = time.time()
        pool.force_cleanup_between_engines("from_key", "to_key")
        cleanup_time = time.time() - start_time
        
        # Should complete reasonably quickly (less than 1 second for 10 engines)
        assert cleanup_time < 1.0, f"Cleanup took too long: {cleanup_time}s for {engine_count} engines"
        
        # All engines should be cleaned up
        assert len(pool.get_engine_stats()) == 0
        
        # All engines should have been unloaded
        for _, engine in engines:
            engine.unload_model.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])