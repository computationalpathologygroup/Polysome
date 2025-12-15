import logging
import threading
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from contextlib import contextmanager
from polysome.engines.base import Engine
import json

logger = logging.getLogger(__name__)


class PoolLockManager:
    """Handles lock timing and monitoring for the engine pool."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._lock_warning_threshold = 1.0  # seconds
    
    @contextmanager
    def timed_lock(self, operation: str, node_id: str):
        """
        Context manager for timed lock acquisition with monitoring.
        
        Args:
            operation: Description of the operation (e.g., "engine acquisition")
            node_id: ID of the requesting node
            
        Yields:
            float: Lock wait time in seconds
        """
        lock_start_time = time.time()
        with self._lock:
            lock_acquired_time = time.time()
            lock_wait_time = lock_acquired_time - lock_start_time
            
            if lock_wait_time > self._lock_warning_threshold:
                logger.warning(
                    f"Node '{node_id}': Long lock wait time for {operation}: {lock_wait_time:.2f}s"
                )
            
            yield lock_wait_time


class EngineLifecycleManager:
    """Handles engine creation and destruction with timeout and error handling."""
    
    def create_engine(
        self, 
        engine_name: str, 
        model_name: str, 
        engine_options: Dict[str, Any],
        node_id: str,
        timeout: Optional[float] = None,
        start_time: Optional[float] = None
    ) -> Engine:
        """
        Create an engine with timeout and error handling.
        
        Args:
            engine_name: Name of the engine
            model_name: Model identifier or path
            engine_options: Engine-specific options
            node_id: ID of the requesting node
            timeout: Maximum time to wait for engine creation
            start_time: Start time for timeout calculation
            
        Returns:
            Created engine instance
            
        Raises:
            TimeoutError: If timeout is exceeded
            RuntimeError: If engine creation fails
        """
        creation_start_time = time.time()
        
        # Check timeout before starting expensive operation
        if timeout is not None and start_time is not None:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Timeout exceeded while acquiring engine '{engine_name}'")
        
        logger.info(
            f"Node '{node_id}': Creating new engine '{engine_name}' for model '{model_name}'"
        )
        
        try:
            from polysome.engines.registry import get_engine
            engine = get_engine(
                engine_name=engine_name,
                model_name=model_name,
                **engine_options
            )
            
            creation_time = time.time() - creation_start_time
            logger.info(
                f"Node '{node_id}': Successfully created engine '{engine_name}' "
                f"for model '{model_name}' (creation time: {creation_time:.2f}s)"
            )
            
            return engine
            
        except Exception as e:
            logger.error(
                f"Node '{node_id}': Failed to create engine '{engine_name}' "
                f"for model '{model_name}': {e}"
            )
            raise RuntimeError(f"Failed to create engine: {e}") from e
    
    def destroy_engine(
        self, 
        engine: Engine, 
        engine_name: str, 
        model_name: str, 
        node_id: str,
        timeout: Optional[float] = None,
        start_time: Optional[float] = None
    ) -> None:
        """
        Destroy an engine with timeout and error handling.
        
        Args:
            engine: Engine instance to destroy
            engine_name: Name of the engine (for logging)
            model_name: Model name (for logging)
            node_id: ID of the requesting node
            timeout: Maximum time to wait for engine destruction
            start_time: Start time for timeout calculation
        """
        unload_start_time = time.time()
        
        # Check timeout before starting potentially blocking operation
        if timeout is not None and start_time is not None:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(
                    f"Node '{node_id}': Timeout exceeded while unloading engine '{engine_name}' "
                    f"for model '{model_name}'"
                )
                return
        
        try:
            engine.unload_model()
            unload_time = time.time() - unload_start_time
            logger.info(
                f"Node '{node_id}': Successfully unloaded engine '{engine_name}' "
                f"for model '{model_name}' (unload time: {unload_time:.2f}s)"
            )
        except Exception as e:
            unload_time = time.time() - unload_start_time
            logger.error(
                f"Node '{node_id}': Error unloading engine '{engine_name}' "
                f"for model '{model_name}' after {unload_time:.2f}s: {e}"
            )


class ReferenceCounter:
    """Thread-safe reference counting with atomic operations."""
    
    def __init__(self):
        self._counts: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def increment(self, key: str) -> int:
        """
        Atomically increment reference count for a key.
        
        Args:
            key: The key to increment
            
        Returns:
            New reference count
        """
        with self._lock:
            current_count = self._counts.get(key, 0)
            new_count = current_count + 1
            self._counts[key] = new_count
            return new_count
    
    def decrement(self, key: str) -> int:
        """
        Atomically decrement reference count for a key.
        
        Args:
            key: The key to decrement
            
        Returns:
            New reference count (minimum 0)
        """
        with self._lock:
            current_count = self._counts.get(key, 0)
            new_count = max(0, current_count - 1)
            if new_count == 0:
                # Remove key when count reaches zero
                self._counts.pop(key, None)
            else:
                self._counts[key] = new_count
            return new_count
    
    def get_count(self, key: str) -> int:
        """
        Get current reference count for a key.
        
        Args:
            key: The key to check
            
        Returns:
            Current reference count (0 if key doesn't exist)
        """
        with self._lock:
            return self._counts.get(key, 0)
    
    def set_count(self, key: str, count: int) -> int:
        """
        Set reference count for a key.
        
        Args:
            key: The key to set
            count: The count to set
            
        Returns:
            The count that was set
        """
        with self._lock:
            if count <= 0:
                self._counts.pop(key, None)
                return 0
            else:
                self._counts[key] = count
                return count
    
    def get_all_counts(self) -> Dict[str, int]:
        """
        Get a snapshot of all current reference counts.
        
        Returns:
            Dictionary of all current counts
        """
        with self._lock:
            return self._counts.copy()
    
    def clear(self) -> None:
        """Clear all reference counts."""
        with self._lock:
            self._counts.clear()


class PoolMetrics:
    """Handles timing, logging, and statistics for the engine pool."""
    
    def log_engine_acquisition(
        self, 
        engine_name: str, 
        model_name: str, 
        node_id: str, 
        ref_count: int,
        lock_wait_time: float,
        reused: bool = True
    ) -> None:
        """
        Log engine acquisition event.
        
        Args:
            engine_name: Name of the engine
            model_name: Model name
            node_id: ID of the requesting node
            ref_count: Current reference count
            lock_wait_time: Time spent waiting for lock
            reused: Whether the engine was reused (True) or created (False)
        """
        action = "Reusing existing" if reused else "Created and cached"
        logger.info(
            f"Node '{node_id}': {action} engine '{engine_name}' for model '{model_name}' "
            f"(ref count: {ref_count}, lock wait: {lock_wait_time:.3f}s)"
        )
    
    def log_engine_release(
        self, 
        engine_name: str, 
        model_name: str, 
        node_id: str, 
        ref_count: int,
        lock_wait_time: float,
        scheduled_for_unload: bool = False,
        cleanup_deferred: bool = False
    ) -> None:
        """
        Log engine release event.
        
        Args:
            engine_name: Name of the engine
            model_name: Model name
            node_id: ID of the requesting node
            ref_count: Current reference count
            lock_wait_time: Time spent waiting for lock
            scheduled_for_unload: Whether engine is scheduled for unloading
            cleanup_deferred: Whether cleanup is deferred
        """
        if scheduled_for_unload:
            logger.info(
                f"Node '{node_id}': Scheduled engine '{engine_name}' for model '{model_name}' "
                f"for unloading (no more references)"
            )
        else:
            status = "cleanup deferred" if cleanup_deferred else ""
            logger.info(
                f"Node '{node_id}': Released engine '{engine_name}' for model '{model_name}' "
                f"(ref count: {ref_count}, {status}, lock wait: {lock_wait_time:.3f}s)".replace(", ,", ",")
            )
    
    def log_engine_creation_started(
        self, 
        engine_name: str, 
        model_name: str, 
        node_id: str,
        lock_wait_time: float
    ) -> None:
        """Log start of engine creation."""
        logger.debug(
            f"Node '{node_id}': Started engine creation for '{engine_name}' "
            f"(lock wait: {lock_wait_time:.3f}s)"
        )
    
    def log_non_existent_engine_release(
        self, 
        engine_name: str, 
        model_name: str, 
        node_id: str,
        lock_wait_time: float
    ) -> None:
        """Log attempt to release non-existent engine."""
        logger.warning(
            f"Node '{node_id}': Attempted to release non-existent engine '{engine_name}' "
            f"for model '{model_name}' (lock wait: {lock_wait_time:.3f}s)"
        )
    
    def log_engine_still_creating(
        self, 
        engine_name: str, 
        model_name: str, 
        node_id: str
    ) -> None:
        """Log attempt to release engine that's still being created."""
        logger.warning(
            f"Node '{node_id}': Attempted to release engine '{engine_name}' "
            f"for model '{model_name}' that is still being created"
        )
    
    def log_pool_cleanup_start(self, engine_count: int) -> None:
        """Log start of pool cleanup."""
        logger.info(f"Cleaning up all engines in pool ({engine_count} engines)")
    
    def log_pool_cleanup_complete(self) -> None:
        """Log completion of pool cleanup."""
        logger.info("Engine pool cleanup complete")
    
    def log_force_unload_engine(
        self, 
        engine_name: str, 
        model_name: str, 
        ref_count: int
    ) -> None:
        """Log force unload of engine during cleanup."""
        logger.info(
            f"Force unloading engine '{engine_name}' "
            f"for model '{model_name}' "
            f"(had {ref_count} references)"
        )
    
    def log_force_unload_error(
        self, 
        engine_name: str, 
        model_name: str, 
        error: Exception
    ) -> None:
        """Log error during force unload."""
        logger.error(
            f"Error during force cleanup of engine '{engine_name}' "
            f"for model '{model_name}': {error}"
        )


@dataclass
class EngineInfo:
    """Information about a shared engine instance."""
    engine: Engine
    engine_name: str
    model_name: str
    engine_options: Dict[str, Any]


class EnginePool:
    """
    Singleton class for managing shared engine instances across nodes.
    
    This class provides efficient engine sharing by:
    - Maintaining a pool of loaded engines with reference counting
    - Only loading engines when first requested
    - Only unloading engines when no nodes are using them
    - Thread-safe operations for concurrent access
    """
    
    _instance: Optional['EnginePool'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'EnginePool':
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the engine pool (only once due to singleton)."""
        if getattr(self, '_initialized', False):
            return
            
        self._engines: Dict[str, EngineInfo] = {}
        self._defer_cleanup = False  # Flag to defer cleanup until workflow end
        self._initialized = True
        
        # Initialize component managers
        self._lock_manager = PoolLockManager()
        self._lifecycle_manager = EngineLifecycleManager()
        self._ref_counter = ReferenceCounter()
        self._metrics = PoolMetrics()
        
        logger.info("EnginePool initialized")
    
    def _generate_engine_key(
        self, 
        engine_name: str, 
        model_name: str, 
        engine_options: Dict[str, Any]
    ) -> str:
        """
        Generate a unique key for an engine configuration.
        
        Args:
            engine_name: Name of the engine (e.g., 'huggingface', 'llama_cpp')
            model_name: Model identifier or path
            engine_options: Engine-specific options
            
        Returns:
            Unique string key for this engine configuration
        """
        # Sort options to ensure consistent key generation
        sorted_options = json.dumps(engine_options, sort_keys=True)
        return f"{engine_name}::{model_name}::{sorted_options}"
    
    def acquire_engine(
        self,
        engine_name: str,
        model_name: str,
        engine_options: Optional[Dict[str, Any]] = None,
        node_id: str = "unknown",
        timeout: Optional[float] = None
    ) -> Engine:
        """
        Acquire an engine instance, loading it if necessary.
        
        Args:
            engine_name: Name of the engine
            model_name: Model identifier or path
            engine_options: Engine-specific options (default: {})
            node_id: ID of the requesting node (for logging)
            timeout: Maximum time to wait for engine acquisition (None for no timeout)
            
        Returns:
            Engine instance ready for use
            
        Raises:
            RuntimeError: If engine creation fails
            TimeoutError: If timeout is exceeded
        """
        if engine_options is None:
            engine_options = {}
            
        engine_key = self._generate_engine_key(engine_name, model_name, engine_options)
        start_time = time.time()
        
        # Try to acquire existing engine
        with self._lock_manager.timed_lock("engine acquisition", node_id) as lock_wait_time:
            if engine_key in self._engines:
                # Engine already exists, increment reference count
                engine_info = self._engines[engine_key]
                if engine_info is not None:  # Not a placeholder
                    ref_count = self._ref_counter.increment(engine_key)
                    self._metrics.log_engine_acquisition(
                        engine_name, model_name, node_id, ref_count, lock_wait_time, reused=True
                    )
                    return engine_info.engine
            
            # Mark that we're creating this engine to prevent duplicate creation
            self._engines[engine_key] = None  # Placeholder
            self._metrics.log_engine_creation_started(engine_name, model_name, node_id, lock_wait_time)
        
        # Engine creation happens outside the lock to prevent deadlock
        try:
            engine = self._lifecycle_manager.create_engine(
                engine_name=engine_name,
                model_name=model_name,
                engine_options=engine_options,
                node_id=node_id,
                timeout=timeout,
                start_time=start_time
            )
            
            # Atomically store the created engine
            with self._lock_manager.timed_lock("engine storage", node_id):
                self._engines[engine_key] = EngineInfo(
                    engine=engine,
                    engine_name=engine_name,
                    model_name=model_name,
                    engine_options=engine_options.copy()
                )
                ref_count = self._ref_counter.set_count(engine_key, 1)
                self._metrics.log_engine_acquisition(
                    engine_name, model_name, node_id, ref_count, 0, reused=False
                )
            
            return engine
            
        except Exception as e:
            # Clean up placeholder on failure
            with self._lock_manager.timed_lock("cleanup", node_id):
                if engine_key in self._engines and self._engines[engine_key] is None:
                    del self._engines[engine_key]
            raise
    
    def set_defer_cleanup(self, defer: bool) -> None:
        """
        Set whether to defer engine cleanup until explicitly called.
        
        Args:
            defer: If True, engines won't be unloaded when reference count hits zero
        """
        with self._lock_manager.timed_lock("set defer cleanup", "system"):
            self._defer_cleanup = defer
            logger.info(f"Engine cleanup deferral set to: {defer}")
    
    def release_engine(
        self,
        engine_name: str,
        model_name: str,
        engine_options: Optional[Dict[str, Any]] = None,
        node_id: str = "unknown",
        timeout: Optional[float] = None
    ) -> None:
        """
        Release an engine instance, unloading it if no longer needed.
        
        Args:
            engine_name: Name of the engine
            model_name: Model identifier or path  
            engine_options: Engine-specific options (default: {})
            node_id: ID of the releasing node (for logging)
            timeout: Maximum time to wait for engine release (None for no timeout)
        """
        if engine_options is None:
            engine_options = {}
            
        engine_key = self._generate_engine_key(engine_name, model_name, engine_options)
        start_time = time.time()
        
        # Atomically decrement reference count and check if cleanup needed
        engine_to_unload = None
        with self._lock_manager.timed_lock("engine release", node_id) as lock_wait_time:
            if engine_key not in self._engines:
                self._metrics.log_non_existent_engine_release(
                    engine_name, model_name, node_id, lock_wait_time
                )
                return
                
            engine_info = self._engines[engine_key]
            if engine_info is None:  # Engine still being created
                self._metrics.log_engine_still_creating(engine_name, model_name, node_id)
                return
                
            ref_count = self._ref_counter.decrement(engine_key)
            
            if ref_count <= 0 and not self._defer_cleanup:
                # No more references and not deferring cleanup, prepare for unload
                engine_to_unload = engine_info.engine
                del self._engines[engine_key]
                self._metrics.log_engine_release(
                    engine_name, model_name, node_id, ref_count, lock_wait_time,
                    scheduled_for_unload=True
                )
            else:
                self._metrics.log_engine_release(
                    engine_name, model_name, node_id, ref_count, lock_wait_time,
                    cleanup_deferred=self._defer_cleanup and ref_count <= 0
                )
        
        # Perform engine unloading outside the lock to prevent deadlock
        if engine_to_unload is not None:
            self._lifecycle_manager.destroy_engine(
                engine=engine_to_unload,
                engine_name=engine_name,
                model_name=model_name,
                node_id=node_id,
                timeout=timeout,
                start_time=start_time
            )
    
    def get_engine_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about currently loaded engines.
        
        Returns:
            Dictionary with engine statistics
        """
        with self._lock_manager.timed_lock("get engine stats", "system"):
            stats = {}
            ref_counts = self._ref_counter.get_all_counts()
            for engine_key, engine_info in self._engines.items():
                if engine_info is not None:  # Skip placeholders
                    stats[engine_key] = {
                        "engine_name": engine_info.engine_name,
                        "model_name": engine_info.model_name,
                        "reference_count": ref_counts.get(engine_key, 0),
                        "engine_options": engine_info.engine_options
                    }
            return stats
    
    def force_cleanup_between_engines(
        self,
        from_engine_key: str,
        to_engine_key: str,
        exclude_keys: Optional[List[str]] = None
    ) -> None:
        """
        Force cleanup of engines between different configurations to prevent OOM.
        This overrides reference counting and immediately unloads engines that don't
        match the target engine configuration.

        Args:
            from_engine_key: Engine key we're transitioning from
            to_engine_key: Engine key we're transitioning to
            exclude_keys: Engine keys to exclude from cleanup (optional)
        """
        if exclude_keys is None:
            exclude_keys = []
        
        logger.info(f"Forcing engine cleanup: {from_engine_key} → {to_engine_key}")
        
        # Find engines that need to be cleaned up
        engines_to_cleanup = []
        with self._lock_manager.timed_lock("force cleanup between engines", "system"):
            ref_counts = self._ref_counter.get_all_counts()
            
            for engine_key, engine_info in list(self._engines.items()):
                if engine_info is None:  # Skip placeholders
                    continue
                
                # For OOM prevention, compare full engine keys (not just base keys)
                # Different configurations of the same model need separate cleanup
                target_key = to_engine_key
                
                # Cleanup engines that don't match target exactly and aren't excluded
                if (engine_key != target_key and 
                    engine_key not in exclude_keys):
                    
                    ref_count = ref_counts.get(engine_key, 0)
                    engines_to_cleanup.append((engine_key, engine_info, ref_count))
                    
                    # Remove from pool immediately to prevent reuse
                    del self._engines[engine_key]
                    self._ref_counter.set_count(engine_key, 0)
                    
                    logger.debug(f"Marked engine for forced cleanup: {engine_key} (had {ref_count} refs)")
        
        # Perform cleanup outside the lock to prevent deadlock
        for engine_key, engine_info, ref_count in engines_to_cleanup:
            try:
                logger.info(f"Force unloading engine: {engine_info.engine_name} (model: {engine_info.model_name})")
                
                # Special handling for vLLM data parallel engines
                if engine_info.engine_name == "vllm_dp":
                    logger.info(f"Gracefully shutting down vLLM_dp engine: {engine_info.model_name}")
                    try:
                        if hasattr(engine_info.engine, 'coordinator') and engine_info.engine.coordinator:
                            engine_info.engine.coordinator.shutdown()
                        else:
                            engine_info.engine.unload_model()
                    except Exception as dp_error:
                        logger.error(f"Error during vLLM_dp shutdown, forcing cleanup: {dp_error}")
                        # Additional force cleanup for vLLM_dp if needed
                        if hasattr(engine_info.engine, 'coordinator'):
                            try:
                                for process in getattr(engine_info.engine.coordinator, 'processes', []):
                                    if process.is_alive():
                                        process.terminate()
                            except Exception as force_error:
                                logger.error(f"Force process termination failed: {force_error}")
                else:
                    # Standard engine cleanup
                    engine_info.engine.unload_model()
                
                logger.info(f"Successfully unloaded engine: {engine_info.engine_name}")
                
            except Exception as e:
                logger.error(f"Error during forced cleanup of {engine_key}: {e}")
        
        if engines_to_cleanup:
            logger.info(f"Forced cleanup complete: unloaded {len(engines_to_cleanup)} engines")
        else:
            logger.debug("No engines needed cleanup")

    def _extract_base_engine_key_from_full_key(self, full_key: str) -> str:
        """
        Extract base engine configuration from full engine key.
        Example: "vllm_dp::model::{'max_model_len': 4096}" -> "vllm_dp::model"
        """
        parts = full_key.split("::")
        if len(parts) >= 2:
            return f"{parts[0]}::{parts[1]}"
        return full_key

    def cleanup_all_engines(self) -> None:
        """
        Force cleanup of all engines in the pool.
        
        This should only be used at the end of workflow execution
        or in emergency situations.
        """
        # First, atomically extract all engines to clean up
        engines_to_cleanup = []
        with self._lock_manager.timed_lock("cleanup all engines", "system"):
            engine_count = len([info for info in self._engines.values() if info is not None])
            self._metrics.log_pool_cleanup_start(engine_count)
            
            ref_counts = self._ref_counter.get_all_counts()
            for engine_key, engine_info in self._engines.items():
                if engine_info is not None:  # Skip placeholders
                    ref_count = ref_counts.get(engine_key, 0)
                    engines_to_cleanup.append((engine_key, engine_info, ref_count))
            
            # Clear the pool immediately
            self._engines.clear()
            self._ref_counter.clear()
        
        # Perform cleanup outside the lock to prevent deadlock
        for engine_key, engine_info, ref_count in engines_to_cleanup:
            try:
                self._metrics.log_force_unload_engine(
                    engine_info.engine_name, engine_info.model_name, ref_count
                )
                # Special handling for data parallel engines that need graceful shutdown
                if engine_info.engine_name == "vllm_dp":
                    logger.info(f"Gracefully shutting down data parallel engine: {engine_info.model_name}")
                    try:
                        engine_info.engine.unload_model()
                    except Exception as dp_error:
                        logger.error(f"Error during data parallel engine shutdown: {dp_error}")
                        # Force cleanup if graceful shutdown fails
                        if hasattr(engine_info.engine, 'coordinator') and engine_info.engine.coordinator:
                            engine_info.engine.coordinator.shutdown()
                else:
                    engine_info.engine.unload_model()
            except Exception as e:
                self._metrics.log_force_unload_error(
                    engine_info.engine_name, engine_info.model_name, e
                )
        
        self._metrics.log_pool_cleanup_complete()
    
    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (primarily for testing).
        
        Warning: This will force cleanup all engines and reset the pool.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.cleanup_all_engines()
                cls._instance = None
                logger.info("EnginePool instance reset")


# Convenience function for getting the global engine pool
def get_engine_pool() -> EnginePool:
    """Get the global engine pool instance."""
    return EnginePool()