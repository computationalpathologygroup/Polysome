# Data Parallelism with vLLM

Data parallelism distributes batch processing across multiple vLLM processes, each running on a different GPU set. Use this for multi-GPU systems to improve throughput.

## Prerequisites

- 4+ CUDA GPUs
- vLLM with data parallelism support (`VLLM_USE_V1=1`)
- Sufficient GPU memory for your model on each GPU set

## Configuration

Set `inference_engine` to `vllm_dp` in your text prompt node:

```json
{
  "id": "my_node",
  "type": "text_prompt",
  "params": {
    "name": "task_name",
    "model_name": "meta-llama/Llama-2-7b-chat-hf",
    "inference_engine": "vllm_dp",
    "engine_options": {
      "data_parallel_size": 2,
      "gpus_per_dp_rank": 2,
      "gpu_memory_utilization": 0.9,
      "max_model_len": 4096
    },
    "batch_size": 32,
    "batch_timeout": 600.0,
    "generation_options": {
      "temperature": 0.7,
      "max_tokens": 512
    }
  }
}
```

## Engine Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_parallel_size` | Number of data parallel ranks | 2 |
| `gpus_per_dp_rank` | GPUs per data parallel rank | 1 |
| `dp_master_ip` | Master IP for coordination | "127.0.0.1" |
| `dp_master_port` | Master port (auto-assigned if None) | None |
| `enable_data_parallel` | Enable/disable data parallelism | true |
| `disable_progress_bars` | Disable vLLM progress bars | true |

Additional vLLM parameters: `trust_remote_code`, `gpu_memory_utilization`, `max_model_len`, `dtype`

## Hardware Configurations

**4 GPUs (2×2)**: 2 ranks × 2 GPUs = GPUs [0,1] and [2,3]
```json
"data_parallel_size": 2, "gpus_per_dp_rank": 2
```

**8 GPUs (4×2)**: 4 ranks × 2 GPUs each
```json
"data_parallel_size": 4, "gpus_per_dp_rank": 2
```

**8 GPUs (2×4)**: 2 ranks × 4 GPUs each (for large models)
```json
"data_parallel_size": 2, "gpus_per_dp_rank": 4
```

## Batch Configuration

- Batch size should be larger than `data_parallel_size` (recommended: 16-64+ items)
- `batch_timeout` (default: 600s) controls maximum wait time for batch completion
- Batches are automatically distributed across ranks (e.g., 33 items with 4 ranks = [9,8,8,8])

## Environment Variables

```bash
export VLLM_USE_V1=1  # Required for data parallelism
export VLLM_LOGGING_LEVEL=INFO  # Optional: for debugging
```

## Complete Example

```json
{
  "name": "data_parallel_workflow",
  "data_dir": "/data",
  "output_dir": "/output",
  "prompts_dir": "/prompts",
  "nodes": [
    {
      "id": "load_data",
      "type": "load",
      "params": {
        "input_data_path": "dataset.jsonl",
        "primary_key": "id"
      },
      "dependencies": []
    },
    {
      "id": "parallel_generation",
      "type": "text_prompt",
      "params": {
        "name": "generate_responses",
        "model_name": "meta-llama/Llama-2-13b-chat-hf",
        "inference_engine": "vllm_dp",
        "engine_options": {
          "data_parallel_size": 4,
          "gpus_per_dp_rank": 2,
          "gpu_memory_utilization": 0.85,
          "dtype": "bfloat16"
        },
        "batch_size": 64,
        "batch_timeout": 600.0
      },
      "dependencies": ["load_data"]
    }
  ]
}
```

## Troubleshooting

**CUDA out of memory**: Reduce `gpu_memory_utilization` or `gpus_per_dp_rank`

**Port conflicts**: Set explicit `dp_master_port` or restart to free ports

**Batch too small**: Increase `batch_size` or reduce `data_parallel_size`

**Worker failures**: Check logs for initialization errors; engine automatically falls back to single-worker mode

## Monitoring

Key log messages:
- `Starting X data parallel workers`
- `Worker rank X is ready`
- `Assigned N prompts to rank X`
- `Received results for batch Y from rank X`

Monitor GPU utilization and memory usage across all GPUs to verify balanced workload distribution.
