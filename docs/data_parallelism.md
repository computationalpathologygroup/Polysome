# Data Parallelism with vLLM in Polysome

This guide explains how to use data parallelism with vLLM to distribute batch processing across multiple GPU sets for improved throughput and better resource utilization in multi-GPU systems.

## Overview

Data parallelism in Polysome allows you to split large batches across multiple vLLM processes, each running on a different set of GPUs. This approach is particularly beneficial when:

- You have multiple GPUs available (4+ GPUs recommended)
- Processing large batches of text (>32 items per batch)
- Want to maximize GPU utilization across your system
- Need higher throughput than single-process vLLM can provide

## Prerequisites

### Hardware Requirements
- **Minimum**: 4 GPUs (2 data parallel ranks × 2 GPUs per rank)
- **Recommended**: 8+ GPUs for optimal performance
- Sufficient GPU memory for your model on each GPU set
- Adequate system RAM for multiprocessing overhead

### Software Requirements
- vLLM installed with data parallelism support (`VLLM_USE_V1=1`)
- CUDA-capable GPUs
- Polysome with data parallel engine support

## Configuration

### Basic Data Parallel Configuration

To use data parallelism, specify the `vllm_dp` engine in your workflow configuration:

```json
{
  "id": "my_text_node",
  "type": "text_prompt_node",
  "params": {
    "name": "parallel_processing",
    "model_name": "meta-llama/Llama-2-7b-chat-hf",
    "inference_engine": "vllm_dp",
    "engine_options": {
      "data_parallel_size": 2,
      "gpus_per_dp_rank": 2,
      "trust_remote_code": true,
      "gpu_memory_utilization": 0.9,
      "max_model_len": 4096
    },
    "batch_size": 32,
    "generation_options": {
      "temperature": 0.7,
      "max_tokens": 512,
      "top_p": 0.9
    }
  }
}
```

### Engine Options for Data Parallelism

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `data_parallel_size` | Number of data parallel ranks | 2 | 4 |
| `gpus_per_dp_rank` | GPUs per data parallel rank | 1 | 2 |
| `dp_master_ip` | Master IP for coordination | "127.0.0.1" | "192.168.1.100" |
| `dp_master_port` | Master port (auto if None) | None | 29500 |
| `enable_data_parallel` | Enable/disable data parallelism | true | false |

Additional vLLM parameters (same as regular vLLM engine):
- `trust_remote_code`: Whether to trust remote model code
- `gpu_memory_utilization`: Fraction of GPU memory to use (0.0-1.0)
- `max_model_len`: Maximum context length
- `dtype`: Model precision ("auto", "float16", "bfloat16")

## Hardware Configuration Examples

### Example 1: 4 GPUs (2×2 Configuration)
```json
"engine_options": {
  "data_parallel_size": 2,
  "gpus_per_dp_rank": 2,
  "gpu_memory_utilization": 0.9
}
```
- **GPU allocation**: Rank 0 uses GPUs 0,1 | Rank 1 uses GPUs 2,3
- **Best for**: Medium-sized models (7B-13B parameters)

### Example 2: 8 GPUs (4×2 Configuration)
```json
"engine_options": {
  "data_parallel_size": 4,
  "gpus_per_dp_rank": 2,
  "gpu_memory_utilization": 0.85
}
```
- **GPU allocation**: 4 ranks, each using 2 GPUs
- **Best for**: Large batches with medium models

### Example 3: 8 GPUs (2×4 Configuration)
```json
"engine_options": {
  "data_parallel_size": 2,
  "gpus_per_dp_rank": 4,
  "gpu_memory_utilization": 0.8
}
```
- **GPU allocation**: 2 ranks, each using 4 GPUs
- **Best for**: Large models (30B+ parameters) requiring more memory per rank

## Performance Tuning

### Batch Size Guidelines

The effectiveness of data parallelism depends heavily on batch size:

| Data Parallel Size | Recommended Batch Size | GPU Memory |
|--------------------|------------------------|------------|
| 2 ranks | 16-64 items | 8-24 GB per rank |
| 4 ranks | 32-128 items | 16-40 GB per rank |
| 8 ranks | 64-256 items | 24+ GB per rank |

### Memory Optimization

1. **GPU Memory Utilization**: Start with 0.9, reduce if you encounter OOM errors
2. **Model Precision**: Use `"dtype": "bfloat16"` for better memory efficiency
3. **Context Length**: Set `max_model_len` to the minimum required for your task

### Batch Distribution

The engine automatically distributes batches across ranks:
- Batch of 33 items with 4 ranks: [9, 8, 8, 8] items per rank
- Handles uneven distributions gracefully
- Maintains output order regardless of processing speed differences

## Environment Setup

### Setting Environment Variables

For manual control or debugging, you can set these environment variables:

```bash
# Enable vLLM v1 API (required for data parallelism)
export VLLM_USE_V1=1

# Optional: Set logging level for debugging
export VLLM_LOGGING_LEVEL=INFO

# Optional: Set specific CUDA devices (handled automatically by engine)
# export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Docker Configuration

When using Docker, ensure all GPUs are accessible:

```bash
docker run --rm --gpus all \
  -e VLLM_USE_V1=1 \
  -v $(pwd)/models:/models \
  -v $(pwd)/workflows:/workflows \
  polysome-runner
```

## Complete Workflow Example

Here's a complete workflow configuration using data parallelism:

```json
{
  "name": "data_parallel_example",
  "data_dir": "/data",
  "output_dir": "/output",
  "prompts_dir": "/prompts",
  "workflow_settings": {
    "optimize_for_engines": true
  },
  "nodes": [
    {
      "id": "load_data",
      "type": "load",
      "params": {
        "name": "load_cases",
        "input_data_path": "large_dataset.jsonl",
        "primary_key": "case_id",
        "output_file_name": "loaded_data.jsonl"
      },
      "dependencies": []
    },
    {
      "id": "parallel_generation",
      "type": "text_prompt_node",
      "params": {
        "name": "generate_responses",
        "model_name": "meta-llama/Llama-2-13b-chat-hf",
        "inference_engine": "vllm_dp",
        "engine_options": {
          "data_parallel_size": 4,
          "gpus_per_dp_rank": 2,
          "trust_remote_code": true,
          "gpu_memory_utilization": 0.85,
          "max_model_len": 4096,
          "dtype": "bfloat16"
        },
        "generation_options": {
          "temperature": 0.7,
          "max_tokens": 512,
          "top_p": 0.9
        },
        "batch_size": 64,
        "system_prompt_file": "system_prompt.txt",
        "user_prompt_file": "user_prompt.txt",
        "use_shared_engines": true,
        "resume": true
      },
      "dependencies": ["load_data"]
    }
  ]
}
```

## Monitoring and Debugging

### Logging

Enable detailed logging to monitor data parallel execution:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Key log messages to watch for:
- `Starting X data parallel workers`
- `Worker rank X is ready`
- `Assigned N prompts to rank X`
- `Received results for batch Y from rank X`

### Performance Monitoring

Monitor these metrics during execution:
- **GPU utilization**: Should be high across all GPUs
- **Memory usage**: Should be consistent across ranks
- **Processing time**: Compare with single-rank baseline
- **Batch distribution**: Verify even work distribution

### Common Issues and Solutions

#### Issue: Workers fail to initialize
```
Error: Worker rank X failed to initialize: CUDA out of memory
```
**Solution**: Reduce `gpu_memory_utilization` or `gpus_per_dp_rank`

#### Issue: Uneven performance across ranks
```
Warning: Long lock wait time for engine acquisition
```
**Solution**: Ensure all GPUs have similar specifications and workloads

#### Issue: Port conflicts
```
Error: Address already in use
```
**Solution**: Set explicit `dp_master_port` or restart processes to free ports

#### Issue: Batch too small for parallelism
```
Info: DP rank 2 needs to process 0 prompts
```
**Solution**: Increase batch size or reduce `data_parallel_size`

## Best Practices

1. **Start Simple**: Begin with 2 ranks and scale up based on performance gains
2. **Monitor Resources**: Watch GPU utilization and memory usage during tuning
3. **Test Thoroughly**: Validate output quality matches single-rank results
4. **Batch Size Matters**: Larger batches generally benefit more from parallelism
5. **Network Considerations**: For multi-node setups, ensure low-latency networking
6. **Fallback Mechanism**: The engine automatically falls back to single-rank mode on failures

## Migration from Single vLLM

To migrate existing workflows from single vLLM to data parallel:

1. Change `inference_engine` from `"vllm"` to `"vllm_dp"`
2. Add data parallel options to `engine_options`
3. Increase `batch_size` to take advantage of parallelism
4. Test with a small subset before full deployment

## Performance Expectations

Typical performance improvements with data parallelism:

- **2 ranks**: 1.5-1.8x throughput improvement
- **4 ranks**: 2.5-3.2x throughput improvement  
- **8 ranks**: 4.0-5.5x throughput improvement

*Actual results depend on model size, batch size, hardware, and workload characteristics.*

## Troubleshooting Checklist

- [ ] vLLM v1 API enabled (`VLLM_USE_V1=1`)
- [ ] Sufficient GPU memory for model + data parallel overhead
- [ ] Batch size appropriate for number of ranks
- [ ] No port conflicts on master port
- [ ] All GPUs accessible and properly configured
- [ ] Network connectivity between ranks (for multi-node setups)
- [ ] Model compatible with data parallelism (most models work)

For additional support, check the vLLM documentation or Polysome issue tracker.