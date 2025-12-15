import logging
import os
import time
import multiprocessing
from typing import List, Dict, Any, TYPE_CHECKING
from multiprocessing import Process, Queue, Manager
from queue import Empty
from polysome.engines.base import Engine

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    from vllm.utils import get_open_port

    VLLM_AVAILABLE = True
except ImportError as e:
    VLLM_AVAILABLE = False
    logging.debug(f"vLLM library not available: {e}")

logger = logging.getLogger(__name__)


class DataParallelLogFormatter(logging.Formatter):
    """Custom logging formatter that prefixes messages with worker rank information."""

    def __init__(self, dp_rank: int = None, fmt: str = None, datefmt: str = None):
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        if datefmt is None:
            datefmt = "%Y-%m-%d %H:%M:%S"
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.dp_rank = dp_rank

    def format(self, record):
        # Add rank prefix to the message
        if self.dp_rank is not None:
            original_msg = record.getMessage()
            if not original_msg.startswith(f"[Rank-{self.dp_rank}]"):
                record.msg = f"[Rank-{self.dp_rank}] {original_msg}"
                record.args = ()
        return super().format(record)


def setup_worker_logging(dp_rank: int, log_level: int = logging.INFO):
    """Set up logging for a data parallel worker process."""
    import os
    import sys

    # Disable vLLM progress bars by setting environment variables
    os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

    # Create a custom formatter for this worker
    formatter = DataParallelLogFormatter(dp_rank=dp_rank)

    # Configure the root logger for this process
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplicate logs
    root_logger.handlers.clear()

    # Create console handler with custom formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level)

    # Configure vLLM logging to reduce noise and disable progress bars
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.WARNING)  # Reduce vLLM noise

    # Disable tqdm progress bars globally for this process
    try:
        import tqdm

        tqdm.tqdm.disable = True
    except ImportError:
        pass

    # Configure other noisy loggers
    for logger_name in ["transformers", "torch", "numpy", "PIL"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    return root_logger


def _vllm_dp_worker(
    dp_rank: int,
    dp_size: int,
    dp_master_ip: str,
    dp_master_port: int,
    gpus_per_dp_rank: int,
    model_name: str,
    vllm_kwargs: Dict[str, Any],
    input_queue: Queue,
    output_queue: Queue,
    error_queue: Queue,
    ready_queue: Queue,
):
    """Worker process for a single data parallel rank."""
    try:
        # Set up logging for this worker process
        setup_worker_logging(dp_rank, log_level=logging.INFO)

        # Log current environment and GPU setup
        logger.info(f"Starting worker process initialization")
        logger.info(
            f"Host CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
        )

        # DON'T set vLLM DP environment variables - we're handling DP ourselves
        # Remove these lines:
        # os.environ["VLLM_DP_RANK"] = str(dp_rank)
        # os.environ["VLLM_DP_SIZE"] = str(dp_size)
        # os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        # os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

        # Parse available GPUs
        host_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not host_cuda_devices:
            raise RuntimeError(
                f"No CUDA_VISIBLE_DEVICES set - cannot determine available GPUs"
            )

        host_gpus = [int(x.strip()) for x in host_cuda_devices.split(",") if x.strip()]

        if len(host_gpus) < (dp_rank + 1) * gpus_per_dp_rank:
            raise RuntimeError(f"Not enough GPUs available!")

        # Calculate which GPUs this rank should use
        start_gpu_idx = dp_rank * gpus_per_dp_rank
        end_gpu_idx = start_gpu_idx + gpus_per_dp_rank
        rank_gpus = host_gpus[start_gpu_idx:end_gpu_idx]

        # CRITICAL: Each rank should ONLY see its assigned GPUs
        visible_devices = ",".join(str(gpu) for gpu in rank_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

        logger.info(f"Physical GPUs assigned to this rank: {rank_gpus}")
        logger.info(f"Set CUDA_VISIBLE_DEVICES={visible_devices}")

        # Set NCCL environment variables to avoid conflicts
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Ensure async CUDA operations

        # Update vllm_kwargs to ensure tensor_parallel_size matches visible GPUs
        vllm_kwargs_copy = vllm_kwargs.copy()
        vllm_kwargs_copy["tensor_parallel_size"] = gpus_per_dp_rank

        # Initialize vLLM engine for this rank
        logger.info(f"Initializing vLLM engine with model {model_name}")
        logger.info(f"vLLM kwargs: {vllm_kwargs_copy}")

        llm = LLM(model=model_name, **vllm_kwargs_copy)

        # Signal that this worker is ready
        ready_queue.put(dp_rank)
        logger.info(f"vLLM engine successfully initialized and ready for processing")

        # Process work items
        while True:
            try:
                work_item = input_queue.get(timeout=1.0)
                if work_item is None:  # Shutdown signal
                    logger.info(f"Received shutdown signal")
                    break

                batch_id, prompts, sampling_params_dict = work_item

                # Convert sampling params dict back to SamplingParams object
                sampling_params = SamplingParams(**sampling_params_dict)

                logger.info(f"Processing batch {batch_id} with {len(prompts)} prompts")

                # Generate responses
                outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

                # Extract generated texts
                results = []
                for output in outputs:
                    if output.outputs:
                        generated_text = output.outputs[0].text
                        results.append(generated_text.strip())
                    else:
                        results.append("Error: No output generated")

                # Send results back
                output_queue.put((batch_id, results))
                logger.info(f"Completed batch {batch_id}")

            except Empty:
                # Timeout waiting for work - this is normal, continue waiting
                continue
            except Exception as e:
                # This is a real error during batch processing
                logger.error(f"Error processing batch: {type(e).__name__}: {e}")
                error_queue.put((dp_rank, f"{type(e).__name__}: {e}"))

    except Exception as e:
        logger.error(f"Fatal error during initialization: {e}")
        error_queue.put((dp_rank, f"Initialization error: {e}"))
    finally:
        # Explicit cleanup before process exit
        try:
            import torch
            import gc
            
            # Clear Python objects
            if 'llm' in locals():
                del llm
            gc.collect()
            
            # Clear CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Worker {dp_rank} completed cleanup")
        except Exception as cleanup_error:
            logger.error(f"Worker {dp_rank} cleanup error: {cleanup_error}")


class DataParallelCoordinator:
    """Manages the lifecycle of data parallel worker processes."""

    def __init__(
        self,
        dp_size: int,
        gpus_per_dp_rank: int,
        dp_master_ip: str = "127.0.0.1",
        dp_master_port: int = None,
    ):
        self.dp_size = dp_size
        self.gpus_per_dp_rank = gpus_per_dp_rank
        self.dp_master_ip = dp_master_ip
        self.dp_master_port = dp_master_port or get_open_port()

        self.processes = []
        self.input_queues = []
        self.manager = Manager()
        self.output_queue = self.manager.Queue()
        self.error_queue = self.manager.Queue()
        self.ready_queue = self.manager.Queue()

        self._is_initialized = False

        # Progress tracking
        self.batch_stats = {
            "total_batches": 0,
            "completed_batches": 0,
            "failed_batches": 0,
            "total_prompts": 0,
            "completed_prompts": 0,
            "start_time": None,
            "batch_start_times": {},
            "batch_completion_times": {},
            "worker_batch_counts": {i: 0 for i in range(dp_size)},
        }

    def start_workers(self, model_name: str, vllm_kwargs: Dict[str, Any]):
        """Start all worker processes."""
        logger.info(f"Starting {self.dp_size} data parallel workers")

        # Validate GPU availability before starting workers
        self._validate_gpu_availability()

        for dp_rank in range(self.dp_size):
            # Create input queue for this worker
            input_queue = self.manager.Queue()
            self.input_queues.append(input_queue)

            # Start worker process
            process = Process(
                target=_vllm_dp_worker,
                args=(
                    dp_rank,
                    self.dp_size,
                    self.dp_master_ip,
                    self.dp_master_port,
                    self.gpus_per_dp_rank,
                    model_name,
                    vllm_kwargs,
                    input_queue,
                    self.output_queue,
                    self.error_queue,
                    self.ready_queue,
                ),
            )
            process.start()
            self.processes.append(process)
            logger.info(
                f"Started worker process for DP rank {dp_rank} (PID: {process.pid})"
            )

        # Wait for all workers to be ready
        ready_workers = set()
        timeout = 600  # 10 minutes timeout for initialization (extended for torch compilation)
        start_time = time.time()

        while len(ready_workers) < self.dp_size:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Timeout waiting for workers to initialize. Ready: {len(ready_workers)}/{self.dp_size}"
                )

            try:
                ready_rank = self.ready_queue.get(timeout=1.0)
                ready_workers.add(ready_rank)
                logger.info(
                    f"Worker rank {ready_rank} is ready ({len(ready_workers)}/{self.dp_size})"
                )
            except:
                # Check for errors
                try:
                    rank, error = self.error_queue.get_nowait()
                    raise RuntimeError(
                        f"Worker rank {rank} failed to initialize: {error}"
                    )
                except:
                    pass  # No errors, continue waiting

        self._is_initialized = True
        logger.info("All data parallel workers are ready")

    def _update_progress_stats(
        self, batch_id: str, dp_rank: int, num_prompts: int, status: str
    ):
        """Update progress statistics for batch processing."""
        current_time = time.time()

        if status == "started":
            self.batch_stats["batch_start_times"][batch_id] = current_time
            self.batch_stats["worker_batch_counts"][dp_rank] += 1
            if self.batch_stats["start_time"] is None:
                self.batch_stats["start_time"] = current_time

        elif status == "completed":
            self.batch_stats["completed_batches"] += 1
            self.batch_stats["completed_prompts"] += num_prompts
            self.batch_stats["batch_completion_times"][batch_id] = current_time

        elif status == "failed":
            self.batch_stats["failed_batches"] += 1

        # Log progress at regular intervals (every 10 batches to reduce log volume)
        if self.batch_stats["completed_batches"] % 10 == 0 or status == "completed":
            self._log_progress_update()

    def _log_progress_update(self):
        """Log a progress update with current statistics."""
        stats = self.batch_stats

        if stats["start_time"] is None:
            return

        elapsed_time = time.time() - stats["start_time"]

        # Calculate throughput
        if elapsed_time > 0:
            batches_per_second = stats["completed_batches"] / elapsed_time
            prompts_per_second = stats["completed_prompts"] / elapsed_time
        else:
            batches_per_second = 0
            prompts_per_second = 0

        # Calculate completion percentage
        if stats["total_batches"] > 0:
            completion_pct = (stats["completed_batches"] / stats["total_batches"]) * 100
        else:
            completion_pct = 0

        # Log concise progress summary
        logger.info(
            f"PROGRESS: {stats['completed_batches']}/{stats['total_batches']} batches "
            f"({completion_pct:.1f}%) | {prompts_per_second:.1f} prompts/sec | "
            f"{len(self._get_healthy_workers())} workers active"
        )

    def _reset_progress_stats(self):
        """Reset progress statistics for a new batch processing session."""
        self.batch_stats = {
            "total_batches": 0,
            "completed_batches": 0,
            "failed_batches": 0,
            "total_prompts": 0,
            "completed_prompts": 0,
            "start_time": None,
            "batch_start_times": {},
            "batch_completion_times": {},
            "worker_batch_counts": {i: 0 for i in range(self.dp_size)},
        }

    def distribute_batch(
        self, prompts: List[str], sampling_params_dict: Dict[str, Any]
    ) -> List[str]:
        """Distribute a batch across workers and collect results with failure recovery."""
        if not self._is_initialized:
            raise RuntimeError("Workers not initialized")

        # Reset progress stats for new batch
        self._reset_progress_stats()

        # Check if we have any healthy workers
        healthy_workers = self._get_healthy_workers()
        if not healthy_workers:
            raise RuntimeError("No healthy workers available")

        # Split prompts across healthy workers only
        prompts_per_worker = len(prompts) // len(healthy_workers)
        remainder = len(prompts) % len(healthy_workers)

        # Initialize progress tracking
        self.batch_stats["total_prompts"] = len(prompts)

        # Distribute work
        batch_assignments = {}
        start_idx = 0

        for i, dp_rank in enumerate(healthy_workers):
            end_idx = start_idx + prompts_per_worker
            if i < remainder:  # Give remainder to first few workers
                end_idx += 1

            if start_idx < len(prompts):
                worker_prompts = prompts[start_idx:end_idx]
                if worker_prompts:  # Only assign if there are prompts
                    batch_id = f"batch_{dp_rank}_{int(time.time() * 1000)}"
                    self.input_queues[dp_rank].put(
                        (batch_id, worker_prompts, sampling_params_dict)
                    )
                    batch_assignments[batch_id] = (
                        dp_rank,
                        start_idx,
                        end_idx,
                        worker_prompts,
                    )

                    # Update progress tracking
                    self.batch_stats["total_batches"] += 1
                    self._update_progress_stats(
                        batch_id, dp_rank, len(worker_prompts), "started"
                    )

                    logger.debug(
                        f"Assigned {len(worker_prompts)} prompts to rank {dp_rank} (batch {batch_id})"
                    )

            start_idx = end_idx

        logger.info(
            f"Distributed {len(prompts)} prompts across {len(batch_assignments)} batches to {len(healthy_workers)} workers"
        )

        # Collect results with failure recovery
        results = [None] * len(prompts)
        completed_batches = 0
        failed_batches = {}

        while completed_batches < len(batch_assignments):
            try:
                batch_id, batch_results = self.output_queue.get(
                    timeout=30
                )  # Shorter timeout for faster failure detection

                if batch_id in batch_assignments:
                    dp_rank, start_idx, end_idx, worker_prompts = batch_assignments[
                        batch_id
                    ]

                    # Place results in correct positions
                    for i, result in enumerate(batch_results):
                        results[start_idx + i] = result

                    completed_batches += 1

                    # Update progress tracking
                    self._update_progress_stats(
                        batch_id, dp_rank, len(worker_prompts), "completed"
                    )

                    logger.debug(
                        f"Received results for batch {batch_id} from rank {dp_rank} "
                        f"({len(worker_prompts)} prompts)"
                    )

            except Exception as e:
                logger.warning(f"Timeout or error collecting results: {e}")

                # Check for worker errors and log detailed health status
                if self._check_worker_errors():
                    self._log_worker_health_status()

                # Check worker health and redistribute failed batches
                healthy_workers = self._get_healthy_workers()
                if not healthy_workers:
                    raise RuntimeError("All workers have failed")

                # Find failed batches and redistribute
                for batch_id, (dp_rank, start_idx, end_idx, worker_prompts) in list(
                    batch_assignments.items()
                ):
                    if dp_rank not in healthy_workers:
                        logger.warning(
                            f"Redistributing failed batch {batch_id} from dead worker {dp_rank}"
                        )

                        # Track failed batch
                        self._update_progress_stats(
                            batch_id, dp_rank, len(worker_prompts), "failed"
                        )

                        # Redistribute to a healthy worker
                        if healthy_workers:
                            new_rank = healthy_workers[0]  # Simple round-robin
                            new_batch_id = (
                                f"batch_{new_rank}_{int(time.time() * 1000)}_retry"
                            )
                            self.input_queues[new_rank].put(
                                (new_batch_id, worker_prompts, sampling_params_dict)
                            )
                            batch_assignments[new_batch_id] = (
                                new_rank,
                                start_idx,
                                end_idx,
                                worker_prompts,
                            )
                            del batch_assignments[batch_id]

                            # Track redistributed batch
                            self._update_progress_stats(
                                new_batch_id, new_rank, len(worker_prompts), "started"
                            )

                            logger.info(
                                f"Redistributed batch to healthy worker {new_rank} as {new_batch_id}"
                            )
                        else:
                            raise RuntimeError(
                                "No healthy workers available for redistribution"
                            )

        # Final progress summary and worker health status
        elapsed_time = (
            time.time() - self.batch_stats["start_time"]
            if self.batch_stats["start_time"]
            else 0
        )
        logger.info(
            f"BATCH PROCESSING COMPLETE: {self.batch_stats['completed_batches']} batches, "
            f"{self.batch_stats['completed_prompts']} prompts in {elapsed_time:.2f}s"
        )
        if self.batch_stats["failed_batches"] > 0:
            logger.warning(f"Failed batches: {self.batch_stats['failed_batches']}")

        # Final worker health status
        self._log_worker_health_status()

        return results

    def _validate_gpu_availability(self):
        """Validate that sufficient GPUs are available for data parallel processing."""
        host_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

        logger.info(f"Validating GPU availability for data parallel processing")
        logger.info(f"Host CUDA_VISIBLE_DEVICES: {host_cuda_devices}")
        logger.info(
            f"Required: {self.dp_size} ranks × {self.gpus_per_dp_rank} GPUs per rank = {self.dp_size * self.gpus_per_dp_rank} total GPUs"
        )

        if not host_cuda_devices:
            raise RuntimeError(
                "No CUDA_VISIBLE_DEVICES set - cannot determine available GPUs"
            )

        try:
            # Parse available GPUs
            host_gpus = [
                int(x.strip()) for x in host_cuda_devices.split(",") if x.strip()
            ]
            available_gpus = len(host_gpus)
            required_gpus = self.dp_size * self.gpus_per_dp_rank

            logger.info(f"Available GPUs: {available_gpus} (physical IDs: {host_gpus})")
            logger.info(f"Required GPUs: {required_gpus}")

            if available_gpus < required_gpus:
                raise RuntimeError(
                    f"Insufficient GPUs available! Need {required_gpus} GPUs "
                    f"({self.dp_size} ranks × {self.gpus_per_dp_rank} GPUs per rank), "
                    f"but only {available_gpus} available: {host_gpus}"
                )

            logger.info("✓ GPU availability validation passed")

            # Log the planned GPU assignment
            logger.info("Planned GPU assignment:")
            for dp_rank in range(self.dp_size):
                start_idx = dp_rank * self.gpus_per_dp_rank
                end_idx = start_idx + self.gpus_per_dp_rank
                assigned_physical_gpus = host_gpus[start_idx:end_idx]
                logical_devices = list(range(self.gpus_per_dp_rank))
                logger.info(
                    f"  Rank {dp_rank}: Physical GPUs {assigned_physical_gpus} → Logical devices {logical_devices}"
                )

        except ValueError as e:
            raise RuntimeError(
                f"Invalid CUDA_VISIBLE_DEVICES format: {host_cuda_devices}"
            ) from e

    def _get_healthy_workers(self) -> List[int]:
        """Get list of healthy worker ranks."""
        healthy_workers = []
        for i, process in enumerate(self.processes):
            if process.is_alive():
                healthy_workers.append(i)
        return healthy_workers

    def _log_worker_health_status(self):
        """Log detailed worker health status."""
        total_workers = len(self.processes)
        healthy_workers = self._get_healthy_workers()
        failed_workers = [i for i in range(total_workers) if i not in healthy_workers]

        logger.info(
            f"WORKER STATUS: {len(healthy_workers)}/{total_workers} workers healthy"
        )

        if healthy_workers:
            logger.info(f"Healthy workers: {healthy_workers}")

        if failed_workers:
            logger.warning(f"Failed workers: {failed_workers}")

        # Log per-worker batch processing stats
        for rank in healthy_workers:
            batch_count = self.batch_stats["worker_batch_counts"].get(rank, 0)
            logger.info(f"Worker {rank}: {batch_count} batches processed")

    def _check_worker_errors(self):
        """Check for and log any worker errors."""
        errors_found = False
        try:
            while True:
                rank, error = self.error_queue.get_nowait()
                logger.error(f"Worker {rank} error: {error}")
                errors_found = True
        except:
            pass  # No more errors in queue

        return errors_found

    def shutdown(self):
        """Shutdown all worker processes with proper VRAM cleanup."""
        logger.info("Shutting down data parallel workers")

        # Send shutdown signals
        for input_queue in self.input_queues:
            input_queue.put(None)

        # Wait for processes to finish with extended timeouts for VRAM cleanup
        for i, process in enumerate(self.processes):
            try:
                # Extended graceful shutdown timeout for large model cleanup
                process.join(timeout=300)
                if process.is_alive():
                    logger.warning(f"Worker {i} did not shutdown gracefully, terminating...")
                    process.terminate()
                    process.join(timeout=30)
                    if process.is_alive():
                        logger.error(f"Worker {i} still alive after terminate, force killing...")
                        process.kill()
                        process.join(timeout=15)
                        if process.is_alive():
                            logger.error(f"Failed to kill worker {i} - zombie process detected")
            except Exception as e:
                logger.error(f"Error shutting down worker {i}: {e}")

        self.processes.clear()
        self.input_queues.clear()
        self._is_initialized = False
        
        # Force CUDA memory cleanup after all workers terminated
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cleared CUDA cache after worker shutdown")
        except Exception as e:
            logger.warning(f"Could not clear CUDA cache: {e}")
        
        logger.info("All workers shut down")


class VLLMDataParallelEngine(Engine):
    """
    Data parallel inference engine using multiple vLLM instances.

    This engine distributes batches across multiple vLLM processes running on different GPU sets,
    enabling better utilization of multi-GPU systems for high-throughput inference.
    """

    AVAILABLE = VLLM_AVAILABLE

    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ):
        """
        Initialize the data parallel vLLM engine.

        Args:
            model_name: The identifier for the model (HF repo ID or path).
            **kwargs: Additional arguments including data parallel configuration:
                - data_parallel_size: Number of data parallel ranks (default: 2)
                - gpus_per_dp_rank: Number of GPUs per rank (default: 1)
                - dp_master_ip: Master IP for coordination (default: "127.0.0.1")
                - dp_master_port: Master port for coordination (default: auto-assigned)
                - enable_data_parallel: Enable/disable data parallelism (default: True)
                - disable_progress_bars: Disable vLLM progress bars (default: True)
                - And other vLLM LLM parameters
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM library not installed. Please install with: pip install vllm"
            )

        super().__init__(model_name)

        # Data parallel configuration
        self.data_parallel_size = kwargs.pop("data_parallel_size", 2)
        self.gpus_per_dp_rank = kwargs.pop("gpus_per_dp_rank", 1)
        self.dp_master_ip = kwargs.pop("dp_master_ip", "127.0.0.1")
        self.dp_master_port = kwargs.pop("dp_master_port", None)
        self.enable_data_parallel = kwargs.pop("enable_data_parallel", True)
        self.disable_progress_bars = kwargs.pop("disable_progress_bars", True)

        # vLLM configuration
        self.vllm_kwargs = {
            "trust_remote_code": True,
            "tensor_parallel_size": self.gpus_per_dp_rank,
            **kwargs,
        }

        # Set global environment variables for progress bar control
        if self.disable_progress_bars:
            os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
            os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

        # Initialize coordinator
        self.coordinator = None

        if self.enable_data_parallel:
            logger.info(
                f"Initializing data parallel vLLM engine with {self.data_parallel_size} ranks, {self.gpus_per_dp_rank} GPUs per rank"
            )
            logger.info(f"Model: {model_name}")
            logger.info(f"vLLM configuration: {self.vllm_kwargs}")
            logger.info(
                f"Data parallel master: {self.dp_master_ip}:{self.dp_master_port}"
            )

            self.coordinator = DataParallelCoordinator(
                dp_size=self.data_parallel_size,
                gpus_per_dp_rank=self.gpus_per_dp_rank,
                dp_master_ip=self.dp_master_ip,
                dp_master_port=self.dp_master_port,
            )

            try:
                self.coordinator.start_workers(model_name, self.vllm_kwargs)
                logger.info(
                    f"✓ Data parallel vLLM engine successfully initialized for model: {model_name}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize data parallel vLLM engine: {e}")
                logger.warning(
                    "Falling back to single worker mode due to initialization failure"
                )
                if self.coordinator:
                    self.coordinator.shutdown()
                    self.coordinator = None
                self.enable_data_parallel = False
                # Initialize single worker fallback
                logger.info("Initializing single worker fallback mode")
                from polysome.engines.vllm import VLLMEngine

                self._single_engine = VLLMEngine(model_name, **self.vllm_kwargs)
        else:
            # Fall back to single process mode
            logger.info("Data parallelism disabled, using single vLLM instance")
            from polysome.engines.vllm import VLLMEngine

            self._single_engine = VLLMEngine(model_name, **self.vllm_kwargs)

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Generate text using the data parallel vLLM engine.

        Args:
            messages: Chat history as list of message dictionaries.
            **kwargs: Generation parameters for SamplingParams.

        Returns:
            The generated text string.
        """
        # For single message, use batch generation with size 1
        results = self.generate_text_batch([messages], **kwargs)
        return results[0] if results else "Error: No output generated"

    def generate_text_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate text for a batch using data parallel processing.

        Args:
            messages_batch: A list of message lists for batch processing.
            **kwargs: Generation parameters for SamplingParams.

        Returns:
            A list of generated text strings.
        """
        if not self.enable_data_parallel or not self.coordinator:
            # Fall back to single engine
            return self._single_engine.generate_text_batch(messages_batch, **kwargs)

        logger.info(
            f"Generating text batch with data parallel vLLM engine for {len(messages_batch)} items"
        )

        try:
            # Check if we have any healthy workers before processing
            healthy_workers = self.coordinator._get_healthy_workers()
            if not healthy_workers:
                logger.warning(
                    "No healthy data parallel workers available, falling back to single worker mode"
                )
                return self._fallback_to_single_worker(messages_batch, **kwargs)

            # Apply chat template to all messages in the batch
            prompts = []
            for messages in messages_batch:
                # Use basic concatenation since we don't have tokenizer loaded in this process
                prompt_string = (
                    "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                    + "\nassistant:"
                )
                prompts.append(prompt_string)

            # Prepare sampling parameters
            sampling_kwargs = {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 16,
                **kwargs,
            }

            # Handle max_new_tokens -> max_tokens conversion
            if "max_new_tokens" in sampling_kwargs:
                sampling_kwargs["max_tokens"] = sampling_kwargs.pop("max_new_tokens")

            # Distribute batch and collect results
            results = self.coordinator.distribute_batch(prompts, sampling_kwargs)

            logger.info(
                f"Data parallel batch generation completed successfully for {len(results)} items"
            )
            return results

        except RuntimeError as e:
            if "workers" in str(e).lower():
                logger.warning(
                    f"Data parallel workers failed: {e}. Falling back to single worker mode"
                )
                return self._fallback_to_single_worker(messages_batch, **kwargs)
            else:
                raise e
        except Exception as e:
            logger.exception(f"Error during data parallel batch text generation: {e}")
            return [f"Error generating text with data parallel vLLM: {e}"] * len(
                messages_batch
            )

    def _fallback_to_single_worker(
        self, messages_batch: List[List[Dict[str, str]]], **kwargs: Any
    ) -> List[str]:
        """Fallback to single worker mode when data parallel workers fail."""
        logger.info("Initializing single worker fallback mode")

        if not hasattr(self, "_single_engine") or not self._single_engine:
            try:
                from polysome.engines.vllm import VLLMEngine

                self._single_engine = VLLMEngine(self.model_name, **self.vllm_kwargs)
                logger.info("Single worker fallback engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize single worker fallback: {e}")
                return [
                    f"Error: Both data parallel and single worker modes failed: {e}"
                ] * len(messages_batch)

        return self._single_engine.generate_text_batch(messages_batch, **kwargs)

    def supports_native_batching(self) -> bool:
        """Data parallel vLLM supports native batch processing."""
        return True

    def unload_model(self) -> None:
        """Unload the data parallel vLLM model and shutdown workers."""
        logger.info(f"Unloading data parallel vLLM model: {self.model_name}")

        if self.coordinator:
            try:
                self.coordinator.shutdown()
                self.coordinator = None
                logger.info(
                    f"Successfully unloaded data parallel vLLM model: {self.model_name}"
                )
            except Exception as e:
                logger.error(f"Error during data parallel vLLM model unload: {e}")

        if hasattr(self, "_single_engine") and self._single_engine:
            try:
                self._single_engine.unload_model()
                self._single_engine = None
            except Exception as e:
                logger.error(f"Error unloading single engine fallback: {e}")


# Example usage for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Example configuration
    model_name = "microsoft/DialoGPT-medium"

    try:
        # Initialize data parallel engine
        engine = VLLMDataParallelEngine(
            model_name=model_name,
            data_parallel_size=2,
            gpus_per_dp_rank=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
        )

        # Test with batch of messages
        test_messages_batch = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of Germany?"},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of Italy?"},
            ],
        ]

        print(f"\n--- Testing Data Parallel vLLM engine with {model_name} ---")
        print(f"Batch size: {len(test_messages_batch)}")

        results = engine.generate_text_batch(
            messages_batch=test_messages_batch,
            temperature=0.7,
            max_tokens=50,
            top_p=0.9,
        )

        for i, result in enumerate(results):
            print(f"\nBatch item {i + 1} result:\n{result}")

        print("\n--- Test Complete ---")

        # Clean up
        engine.unload_model()

    except Exception as e:
        logging.exception(f"Error during data parallel vLLM engine test: {e}")
        print("\n--- Test Failed ---")

